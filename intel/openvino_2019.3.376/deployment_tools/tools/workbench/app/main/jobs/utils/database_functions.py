"""
 OpenVINO Profiler
 Common functions for work with the database

 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from app.main.models.base_model import BaseModel
from app.main.models.enumerates import StatusEnum, STATUS_PRIORITY
from app.main.models.factory import write_record


def set_status_in_db(table: type(BaseModel), item_id: int, status: StatusEnum,
                     session, message: str = None, force: bool = False):
    record = session.query(table).get(item_id)
    if record and (force or STATUS_PRIORITY[record.status] < STATUS_PRIORITY[status]):
        record.status = status
        if message:
            record.error_message = message
        write_record(record, session)
