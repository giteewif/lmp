# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

# Adapted from https://github.com/vllm-project/vllm/blob/0ce0539d4750f9ebcd9b19d7085ca3b934b9ec67/vllm/logger.py
"""Logging configuration for lpllm from sllm_store."""

import logging
import os
import sys

_FORMAT = "%(levelname)s %(asctime)s.%(msecs)03d%(nsecs)03d %(filename)s:%(lineno)d] %(message)s"
# We'll use a custom formatter to add ns, since logging.Formatter does not natively support nanoseconds.
# _DATE_FORMAT provides up to the seconds; ms and ns are appended in the formatter.
_DATE_FORMAT = "%m-%d %H:%M:%S"

import time

class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages and appends ns."""

    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)

    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        t = time.strftime(self.datefmt, ct)
        msecs = int(record.msecs)
        # Try to get high-res timestamp for ns; fallback to '000'
        # Python 3.7+: record.relativeCreated has microseconds but we want ns.
        # We'll get ns from the floating timestamp
        ns = int((record.created - int(record.created)) * 1e9) % 1000
        return f"{t}.{msecs:03d}{ns:03d}"

    def format(self, record):
        # Add a field for nanoseconds for use in format string.
        msecs = int(record.msecs)
        ns = int((record.created - int(record.created)) * 1e9) % 1000
        record.msecs = msecs
        record.nsecs = ns
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("lpllm")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler

    _default_handler = logging.StreamHandler(sys.stdout)
    _default_handler.flush = sys.stdout.flush  # type: ignore
    _default_handler.setLevel(logging.DEBUG)
    _root_logger.addHandler(_default_handler)

    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    # Setting this will avoid the message
    # being propagated to the parent logger.
    _root_logger.propagate = False


_setup_logger()


def init_logger(name: str):
    # Use the same settings as above for root logger
    logger = logging.getLogger(name)
    # INFO, DEBUG, WARNING
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(log_level)

    # Ensure the handler's level matches the logger's level
    if _default_handler:
        _default_handler.setLevel(log_level)

    logger.addHandler(_default_handler)
    logger.propagate = False

    return logger
