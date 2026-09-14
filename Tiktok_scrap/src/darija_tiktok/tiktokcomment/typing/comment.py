import json

from datetime import datetime

from typing import Optional, List, Dict, Any

class Comment:
    def __init__(
        self: 'Comment',
        comment_id: str,
        username: str,
        nickname: str,
        comment: str,
        create_time: Any,
        avatar: str,
        total_reply: int,
        replies: Optional[List['Comment']] = None
    ) -> None:
        self._comment_id: str = str(comment_id) if comment_id is not None else ""
        self._username: str = username or ""
        self._nickname: str = nickname or ""
        self._comment: str = comment or ""
        if isinstance(create_time, (int, float)):
            self._create_time: str = datetime.fromtimestamp(create_time).strftime("%Y-%m-%dT%H:%M:%S")
        elif isinstance(create_time, str) and create_time.isdigit():
            self._create_time: str = datetime.fromtimestamp(int(create_time)).strftime("%Y-%m-%dT%H:%M:%S")
        else:
            self._create_time: str = str(create_time) if create_time else ""
        self._avatar: str = avatar or ""
        self._total_reply: int = int(total_reply or 0)
        self._replies: List['Comment'] = replies if replies is not None else []

    @property
    def comment_id(
        self: 'Comment'
    ) -> str:
        return self._comment_id
    
    @property
    def username(
        self: 'Comment'
    ) -> str:
        return self._username
    
    @property
    def nickname(
        self: 'Comment'
    ) -> str:
        return self._nickname
    
    @property
    def comment(
        self: 'Comment'
    ) -> str:
        return self._comment
    
    @property
    def create_time(
        self: 'Comment'
    ) -> str:
        return self._create_time
    
    @property
    def avatar(
        self: 'Comment'
    ) -> str:
        return self._avatar
    
    @property
    def total_reply(
        self: 'Comment'
    ) -> int:
        return self._total_reply
    
    @property
    def replies(
        self: 'Comment'
    ) -> List['Comment']:
        return self._replies
    
    @property
    def dict(
        self: 'Comment'
    ) -> Dict[str, Any]:
        return {
            'comment_id': self._comment_id,
            'username': self._username,
            'nickname': self._nickname,
            'comment': self._comment,
            'create_time': self._create_time,
            'avatar': self._avatar,
            'total_reply': self._total_reply,
            'replies': [reply.dict for reply in self._replies]
        }
    
    @property
    def json(
        self: 'Comment'
    ) -> str:
        return json.dumps(self.dict, ensure_ascii=False)
    
    def __str__(
        self: 'Comment'
    ) -> str:
        return self.json

    def __repr__(
        self: 'Comment'
    ) -> str:
        return f"<Comment id={self._comment_id} user={self._username}>"