from pydantic import BaseModel
from typing import List, Any, Optional, Union

class DeviceParams(BaseModel):
    id: Optional[str] = None
    custom_code: Optional[str] = None
    connection_type: Optional[str] = None
    rtsp_url: Optional[str] = None
    device: Optional[Union[str, int]] = None
    lanes: Optional[List[Any]] = None
    props: Optional[Any] = None
    effects: Optional[Any] = None
