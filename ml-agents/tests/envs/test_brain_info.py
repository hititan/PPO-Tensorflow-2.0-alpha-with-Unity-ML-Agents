import unittest.mock as mock
import pytest
import struct

import numpy as np

from mlagents.envs import UnityEnvironment, UnityEnvironmentException, UnityActionException, \
    BrainInfo
from tests.mock_communicator import MockCommunicator

