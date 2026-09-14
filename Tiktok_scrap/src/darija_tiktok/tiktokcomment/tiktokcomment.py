from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional
from requests import Session, Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
import jmespath

from .typing import Comments, Comment


class TiktokComment:
    BASE_URL: str = 'https://www.tiktok.com'
    API_URL: str = f'{BASE_URL}/api'

    def __init__(
        self,
        cookies: Optional[Dict[str, str]] = None,
        reply_workers: int = 6,
        include_replies: bool = True,
    ) -> None:
        self.__session: Session = Session()
        
        # HTTP Connection Pooling & Keep-Alive
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(
            pool_connections=50,
            pool_maxsize=50,
            max_retries=retry_strategy
        )
        self.__session.mount('https://', adapter)
        self.__session.mount('http://', adapter)

        self.__session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            ),
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.tiktok.com/',
            'Origin': 'https://www.tiktok.com',
        })
        if cookies:
            self.__session.cookies.update(cookies)
        self.aweme_id: str = ""
        self.reply_workers: int = reply_workers
        self.include_replies: bool = include_replies

    def __extract_comment_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fast field extraction from TikTok comment JSON."""
        return jmespath.search(
            """
            {
                comment_id: cid,
                username: user.unique_id,
                nickname: user.nickname,
                comment: text,
                create_time: create_time,
                avatar: user.avatar_thumb.url_list[0],
                total_reply: reply_comment_total
            }
            """,
            data
        ) or {}

    def get_all_replies(
        self,
        comment_id: str,
        aweme_id: Optional[str] = None
    ) -> List[Comment]:
        """Fetch all nested replies for a specific parent comment."""
        all_replies: List[Comment] = []
        page: int = 1
        video_id = aweme_id or self.aweme_id
        while True:
            replies = self.get_replies(
                comment_id=comment_id,
                aweme_id=video_id,
                page=page
            )
            if not replies:
                break
            all_replies.extend(replies)
            if len(replies) < 50:
                break
            page += 1
        return all_replies

    def get_replies(
        self,
        comment_id: str,
        aweme_id: Optional[str] = None,
        size: int = 50,
        page: int = 1
    ) -> List[Comment]:
        video_id = aweme_id or self.aweme_id
        try:
            response: Response = self.__session.get(
                f'{self.API_URL}/comment/list/reply/',
                params={
                    'aid': 1988,
                    'comment_id': comment_id,
                    'item_id': video_id,
                    'count': size,
                    'cursor': (page - 1) * size
                },
                timeout=12
            )
            if response.status_code != 200:
                return []
            res_data = response.json()
            raw_comments = res_data.get('comments') or []
            results = []
            for comment in raw_comments:
                if isinstance(comment, dict):
                    extracted = self.__extract_comment_dict(comment)
                    results.append(Comment(**extracted, replies=[]))
            return results
        except Exception as exc:
            logger.debug(f"Failed to get replies for comment {comment_id}: {exc}")
            return []

    def _parse_comments_batch(
        self,
        raw_comments: List[Dict[str, Any]],
        aweme_id: str
    ) -> List[Comment]:
        """Parse a batch of top-level comments and fetch replies in parallel."""
        parsed_items: List[Dict[str, Any]] = []
        comments_with_replies: List[Tuple[int, str]] = []

        for idx, raw in enumerate(raw_comments):
            if not isinstance(raw, dict):
                continue
            extracted = self.__extract_comment_dict(raw)
            cid = str(extracted.get('comment_id') or raw.get('cid', ''))
            total_reply = int(extracted.get('total_reply') or 0)
            parsed_items.append({
                'index': idx,
                'extracted': extracted,
                'cid': cid,
                'total_reply': total_reply,
                'replies': []
            })
            if self.include_replies and total_reply > 0 and cid:
                comments_with_replies.append((idx, cid))

        # Parallel reply fan-out
        if comments_with_replies and self.reply_workers > 1:
            with ThreadPoolExecutor(max_workers=min(self.reply_workers, len(comments_with_replies))) as executor:
                future_to_idx = {
                    executor.submit(self.get_all_replies, cid, aweme_id): idx
                    for idx, cid in comments_with_replies
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        replies = future.result()
                        for item in parsed_items:
                            if item['index'] == idx:
                                item['replies'] = replies
                                break
                    except Exception as e:
                        logger.debug(f"Error in parallel reply worker: {e}")
        elif comments_with_replies:
            for idx, cid in comments_with_replies:
                try:
                    replies = self.get_all_replies(cid, aweme_id)
                    for item in parsed_items:
                        if item['index'] == idx:
                            item['replies'] = replies
                            break
                except Exception as e:
                    logger.debug(f"Error fetching replies: {e}")

        # Construct final Comment objects
        return [
            Comment(**item['extracted'], replies=item['replies'])
            for item in parsed_items
        ]

    def get_comments(
        self,
        aweme_id: str,
        size: int = 50,
        page: int = 1,
        cursor: Optional[int] = None,
    ) -> Comments:
        self.aweme_id = str(aweme_id)
        current_cursor = cursor if cursor is not None else (page - 1) * size

        try:
            response: Response = self.__session.get(
                f'{self.API_URL}/comment/list/',
                params={
                    'aid': 1988,
                    'aweme_id': self.aweme_id,
                    'count': size,
                    'cursor': current_cursor
                },
                timeout=12
            )
            response.raise_for_status()
            response_data = response.json()
        except Exception as exc:
            logger.debug(f"Error requesting comments for aweme_id={aweme_id} cursor={current_cursor}: {exc}")
            response_data = {}

        raw_comments = response_data.get('comments') or []
        caption = ""
        video_url = ""

        if raw_comments and isinstance(raw_comments, list):
            first = raw_comments[0]
            if isinstance(first, dict):
                share_info = first.get('share_info') or {}
                caption = share_info.get('title') or ""
                video_url = share_info.get('url') or ""

        parsed_comments = self._parse_comments_batch(raw_comments, self.aweme_id)

        return Comments(
            caption=caption,
            video_url=video_url,
            comments=parsed_comments,
            has_more=int(response_data.get('has_more', 0) or 0),
            cursor=int(response_data.get('cursor', 0) or 0),
            total=int(response_data.get('total', 0) or 0),
        )

    def get_all_comments(
        self,
        aweme_id: str,
        max_comments: Optional[int] = None,
        delay: float = 0.05
    ) -> Comments:
        self.aweme_id = str(aweme_id)
        data: Comments = self.get_comments(aweme_id=self.aweme_id, cursor=0)
        seen_ids = {
            str(comment.comment_id)
            for comment in data.comments
            if comment.comment_id
        }
        cursor = data.cursor
        total = data.total

        while (cursor < total or data.has_more) and (max_comments is None or len(data.comments) < max_comments):
            if delay > 0:
                time.sleep(delay)
            more_comments = self.get_comments(aweme_id=self.aweme_id, cursor=cursor)
            if not more_comments.comments:
                break

            new_found = 0
            for comment in more_comments.comments:
                cid = str(comment.comment_id) if comment.comment_id else None
                if cid and cid in seen_ids:
                    continue
                if cid:
                    seen_ids.add(cid)
                data.comments.append(comment)
                new_found += 1
                if max_comments and len(data.comments) >= max_comments:
                    break

            if new_found == 0 or more_comments.cursor <= cursor:
                break

            cursor = more_comments.cursor
            total = max(total, more_comments.total)
            data._cursor = cursor
            data._total = total
            data._has_more = more_comments.has_more

        return data

    def __call__(
        self,
        aweme_id: str,
        max_comments: Optional[int] = None
    ) -> Comments:
        return self.get_all_comments(
            aweme_id=aweme_id,
            max_comments=max_comments
        )
