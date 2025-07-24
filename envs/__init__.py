from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

try:
    smac = True
    from .smac_v1 import StarCraft2EnvWrapper
except Exception as e:
    print(e)
    smac = False

try:
    smacv2 = True
    from .smac_v2 import StarCraft2Env2Wrapper
except Exception as e:
    print(e)
    smacv2 = False

# try:
#     smacv2_10_8 = True
#     from .smac_v2_10_8 import StarCraftCapabilityEnvWrapper
# except Exception as e:
#     print(e)
#     smacv2_10_8 = False


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}

if smac:
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2EnvWrapper)
    if sys.platform == "linux":
        os.environ["SC2PATH"] = os.path.join(os.path.expandvars("$HOME"), "StarCraftII")
else:
    print("SMAC V1 is not supported...")

if smacv2:
    REGISTRY["sc2_v2"] = partial(env_fn, env=StarCraft2Env2Wrapper)
    if sys.platform == "linux":
        os.environ["SC2PATH"] = os.path.join(os.path.expandvars("$HOME"), "StarCraftII")
else:
    print("SMAC V2 is not supported...")

# if smacv2_10_8:
#     REGISTRY["sc2_v2_10_8"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
#     if sys.platform == "linux":
#         os.environ["SC2PATH"] = os.path.join(os.path.expandvars("$HOME"), "StarCraftII")
# else:
#     print("SMAC V2 10.8 is not supported...")

print("Supported environments:", REGISTRY)
