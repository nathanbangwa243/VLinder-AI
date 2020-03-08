#!/bin/bash
# Copyright (c) 2018-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


source $INSTALLDIR/bin/setupvars.sh

export WB_LOG_LEVEL=DEBUG
export WB_LOG_FILE=${OPENVINO_WORKBENCH_ROOT}/server.log
export API_PORT=5666
export PROXY_PORT=${PORT:-5665}

sudo service postgresql start

sudo -E ./start_rabbitmq.sh

celery -A app.main.tasks.task worker --loglevel=${WB_LOG_LEVEL} \
                                     -f ${WB_LOG_FILE} &

./wait_until_db_is_live.sh

gunicorn --worker-class eventlet -w 1 -b 127.0.0.1:${API_PORT} workbench:APP  \
                                                    --log-level ${WB_LOG_LEVEL} \
                                                    --log-file ${WB_LOG_FILE} --error-logfile ${WB_LOG_FILE} \
                                                    --capture-output --enable-stdio-inheritance &

cd proxy
source "$NVM_DIR/nvm.sh" && npm start
