import json
import os
from functools import partial

from metasim.cfg.checkers import BaseChecker
from metasim.cfg.objects import NonConvexRigidObjCfg, RigidObjCfg
from metasim.cfg.tasks.base_task_cfg import BaseTaskCfg
from metasim.constants import BenchmarkType, TaskType
from metasim.utils import configclass

config_filepath = "roboverse_data/trajs/simpler_env/MoveNearGoogleInScene-v0/task_config.json"

object_config_dict = {
    "baked_opened_coke_can_v2": RigidObjCfg(
        name="baked_opened_coke_can_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/coke_can/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_opened_redbull_can_v2": RigidObjCfg(
        name="baked_opened_redbull_can_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/redbull_can/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_apple_v2": RigidObjCfg(
        name="baked_apple_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/apple/mobility.urdf",
        fix_base_link=False,
    ),
    "blue_plastic_bottle": RigidObjCfg(
        name="blue_plastic_bottle",
        urdf_path="roboverse_data/assets/simpler_env/models/blue_plastic_bottle/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_opened_pepsi_can_v2": RigidObjCfg(
        name="baked_opened_pepsi_can_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/pepsi_can/mobility.urdf",
        fix_base_link=False,
    ),
    "orange": RigidObjCfg(
        name="orange",
        urdf_path="roboverse_data/assets/simpler_env/models/orange/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_opened_7up_can_v2": RigidObjCfg(
        name="baked_opened_7up_can_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/7up_can/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_opened_soda_can_v2": RigidObjCfg(
        name="baked_opened_soda_can_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/soda_can/mobility.urdf",
        fix_base_link=False,
    ),
    "baked_sponge_v2": RigidObjCfg(
        name="baked_sponge_v2",
        urdf_path="roboverse_data/assets/simpler_env/models/sponge/mobility.urdf",
        fix_base_link=False,
    ),
}


@configclass
class SimplerEnvMoveNearCfg(BaseTaskCfg):
    # source_benchmark = BenchmarkType.SIMPLERENVGRASPSINGLEOPENEDCOKECAN
    # task_type = TaskType.TABLETOP_MANIPULATION

    # episode_length = 200

    # objects = [
    #     RigidObjCfg(
    #         name="opened_coke_can", urdf_path="assets/simpler_env/models/coke_can/mobility.urdf", fix_base_link=False
    #     ),
    #     NonConvexRigidObjCfg(
    #         name="scene",
    #         usd_path="assets/simpler_env/scenes/google_pick_coke_can_1_v4/google_pick_coke_can_1_v4.glb",
    #         urdf_path="assets/simpler_env/scenes/google_pick_coke_can_1_v4/mobility.urdf",
    #         fix_base_link=True,
    #         mesh_pose=[0, 0, 0, 0.707, 0.707, 0, 0],
    #     ),
    # ]
    # traj_filepath = MISSING
    # checker = BaseChecker()

    def __init__(self, subtask_id=0):
        # load json file
        with open(config_filepath) as f:
            all_config_dict = json.load(f)
            assert subtask_id < len(all_config_dict), f"subtask_id {subtask_id} out of range"
            self.config_dict = all_config_dict[subtask_id]
        self.source_benchmark = BenchmarkType.SIMPLERENVMOVENEAR
        self.task_type = TaskType.TABLETOP_MANIPULATION
        self.episode_length = 200
        self.objects = [
            object_config_dict[self.config_dict["extras"]["triplet"][0]],
            object_config_dict[self.config_dict["extras"]["triplet"][1]],
            object_config_dict[self.config_dict["extras"]["triplet"][2]],
            NonConvexRigidObjCfg(
                name="scene",
                usd_path="roboverse_data/assets/simpler_env/scenes/google_pick_coke_can_1_v4/google_pick_coke_can_1_v4.glb",
                urdf_path="roboverse_data/assets/simpler_env/scenes/google_pick_coke_can_1_v4/mobility.urdf",
                fix_base_link=True,
                mesh_pose=[0, 0, 0, 0.707, 0.707, 0, 0],
            ),
        ]
        self.traj_filepath = os.path.join(
            "roboverse_data/trajs/simpler_env/MoveNearGoogleInScene-v0", self.config_dict["path"]
        )
        self.checker = BaseChecker()
        self.decimation = 20


SimplerEnvMoveNear0Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=0)
SimplerEnvMoveNear1Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=1)
SimplerEnvMoveNear2Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=2)
SimplerEnvMoveNear3Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=3)
SimplerEnvMoveNear4Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=4)
SimplerEnvMoveNear5Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=5)
SimplerEnvMoveNear6Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=6)
SimplerEnvMoveNear7Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=7)
SimplerEnvMoveNear8Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=8)
SimplerEnvMoveNear9Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=9)
SimplerEnvMoveNear10Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=10)
SimplerEnvMoveNear11Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=11)
SimplerEnvMoveNear12Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=12)
SimplerEnvMoveNear13Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=13)
SimplerEnvMoveNear14Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=14)
SimplerEnvMoveNear15Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=15)
SimplerEnvMoveNear16Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=16)
SimplerEnvMoveNear17Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=17)
SimplerEnvMoveNear18Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=18)
SimplerEnvMoveNear19Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=19)
SimplerEnvMoveNear20Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=20)
SimplerEnvMoveNear21Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=21)
SimplerEnvMoveNear22Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=22)
SimplerEnvMoveNear23Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=23)
SimplerEnvMoveNear24Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=24)
SimplerEnvMoveNear25Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=25)
SimplerEnvMoveNear26Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=26)
SimplerEnvMoveNear27Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=27)
SimplerEnvMoveNear28Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=28)
SimplerEnvMoveNear29Cfg = partial(SimplerEnvMoveNearCfg, subtask_id=29)
