import json

from typing import List, Any, Dict, Optional

from .comment import Comment

class Comments:
    def __init__(
        self: 'Comments',
        caption: Optional[str] = "",
        video_url: Optional[str] = "",
        comments: Optional[List[Comment]] = None,
        has_more: int = 0,
        cursor: int = 0,
        total: int = 0,
    ) -> None:
        self._caption: str = caption or ""
        self._video_url: str = video_url or ""
        self._comments: List[Comment] = comments if comments is not None else []
        self._has_more: int = int(has_more or 0)
        self._cursor: int = int(cursor or 0)
        self._total: int = int(total or 0)

    @property
    def caption(
        self: 'Comments'
    ) -> str:
        return self._caption
    
    @property
    def video_url(
        self: 'Comments'
    ) -> str:
        return self._video_url
    
    @property
    def comments(
        self: 'Comments'
    ) -> List[Comment]:
        return self._comments
    
    @property
    def has_more(
        self: 'Comments'
    ) -> int:
        return self._has_more

    @property
    def cursor(
        self: 'Comments'
    ) -> int:
        return self._cursor

    @property
    def total(
        self: 'Comments'
    ) -> int:
        return self._total
    
    @property
    def dict(
        self: 'Comments'
    ) -> Dict[str, Any]:
        return {
            'caption': self._caption,
            'video_url': self._video_url,
            'comments': [comment.dict for comment in self._comments],
            'has_more': self._has_more,
            'cursor': self._cursor,
            'total': self._total,
        }
    
    @property
    def json(
        self: 'Comments'
    ) -> str:
        return json.dumps(self.dict, ensure_ascii=False)
    
    def __str__(
        self: 'Comments'
    ) -> str:
        return self.json

    def __repr__(
        self: 'Comments'
    ) -> str:
        return f"<Comments count={len(self._comments)} total={self._total}>"