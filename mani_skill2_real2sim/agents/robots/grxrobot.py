import numpy as np
import sapien.core as sapien
from sapien.core import Pose

from mani_skill2_real2sim.agents.base_agent import BaseAgent
from mani_skill2_real2sim.agents.configs.grx_robot import defaults
from mani_skill2_real2sim.utils.common import compute_angle_between
from mani_skill2_real2sim.utils.sapien_utils import (
    get_entity_by_name,
    get_pairwise_contact_impulse,
)


class GrxRobot(BaseAgent):
    # TODO
    _config: defaults.GrxDefaultConfig

    """
        WidowX250 6DoF robot
        links:
            [Actor(name="base_link", id="2"), Actor(name="shoulder_link", id="3"), Actor(name="upper_arm_link", id="4"), Actor(name="upper_forearm_link", id="5"), 
            Actor(name="lower_forearm_link", id="6"), Actor(name="wrist_link", id="7"), Actor(name="gripper_link", id="8"), Actor(name="ee_arm_link", id="9"), 
            Actor(name="gripper_prop_link", id="15"), Actor(name="gripper_bar_link", id="10"), Actor(name="fingers_link", id="11"), 
            Actor(name="left_finger_link", id="14"), Actor(name="right_finger_link", id="13"), Actor(name="ee_gripper_link", id="12")]
        active_joints: 
            ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'left_finger', 'right_finger']
        joint_limits:
            [[-3.1415927  3.1415927]
            [-1.8849556  1.9896754]
            [-2.146755   1.6057029]
            [-3.1415827  3.1415827]
            [-1.7453293  2.146755 ]
            [-3.1415827  3.1415827]
            [ 0.015      0.037    ]
            [ 0.015      0.037    ]]
    """

    @classmethod
    def get_default_config(cls):
        return defaults.GrxDefaultConfig()

    def __init__(
        self, scene, control_freq, control_mode=None, fix_root_link=True, config=None
    ):
        if control_mode is None:  # if user did not specify a control_mode
            control_mode = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
        super().__init__(
            scene,
            control_freq,
            control_mode=control_mode,
            fix_root_link=fix_root_link,
            config=config,
        )

    def _after_init(self):
        super()._after_init()
        
        # ignore collision between gripper bar link and two gripper fingers
        # gripper_bar_link = get_entity_by_name(self.robot.get_links(), "gripper_bar_link")
        # left_finger_link = get_entity_by_name(self.robot.get_links(), "left_finger_link")
        # right_finger_link = get_entity_by_name(self.robot.get_links(), "right_finger_link")
        # for l in gripper_bar_link.get_collision_shapes():
        #     l.set_collision_groups(1, 1, 0b11, 0)
        # for l in left_finger_link.get_collision_shapes():
        #     l.set_collision_groups(1, 1, 0b01, 0)
        # for l in right_finger_link.get_collision_shapes():
        #     l.set_collision_groups(1, 1, 0b10, 0)
        
        self.base_link = [x for x in self.robot.get_links() if x.name == "base_link"][0]


        self.left_jaw_joint = get_entity_by_name(
            self.robot.get_joints(), "left_jaw_joint"
        )
        self.right_jaw_joint = get_entity_by_name(
            self.robot.get_joints(), "right_jaw_joint"
        )

        self.left_hand_pitch_link = get_entity_by_name(
            self.robot.get_links(), "left_hand_pitch_link"
        )
        self.left_jaw_link = get_entity_by_name(
            self.robot.get_links(), "left_jaw_link"
        )
        self.right_hand_pitch_link = get_entity_by_name(
            self.robot.get_links(), "right_hand_pitch_link"
        )
        self.right_jaw_link = get_entity_by_name(
            self.robot.get_links(), "right_jaw_link"
        )

    # 1代表全部闭合，计算的是左右手的jaw关节的开闭状态
    def get_gripper_closedness(self):
        finger_qpos = self.robot.get_qpos()[[10, 18]]
        finger_qlim = self.robot.get_qlimits()[[10, 18]]
        closedness_left = (finger_qlim[0, 1] - finger_qpos[0]) / (
            finger_qlim[0, 1] - finger_qlim[0, 0]
        )
        closedness_right = (finger_qpos[1] - finger_qlim[1, 0]) / (
            finger_qlim[1, 1] - finger_qlim[1, 0]
        )
        # return np.maximum(np.mean([closedness_left, closedness_right]), 0.0)
        return closedness_left,closedness_right

    def get_fingers_info(self):

        left_jaw_pos = self.left_jaw_link.get_pose().p
        right_jaw_pos = self.right_jaw_link.get_pose().p

        left_jaw_vel = self.left_jaw_link.get_velocity()
        right_jaw_vel = self.right_jaw_link.get_velocity()

        return {
            "left_jaw_pos": left_jaw_pos,
            "right_jaw_pos": right_jaw_pos,
            "left_jaw_vel": left_jaw_vel,
            "right_jaw_vel": right_jaw_vel,
        }

    def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=60):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse_jaw = get_pairwise_contact_impulse(
            contacts, self.left_jaw_link, actor
        )
        limpulse_hand = get_pairwise_contact_impulse(
            contacts, self.left_hand_pitch_link, actor
        )

        # direction to open the gripper
        ldirection_jaw = self.left_jaw_link.pose.to_transformation_matrix()[:3, 1]
        ldirection_hand = self.left_hand_pitch_link.pose.to_transformation_matrix()[:3, 1]

        # angle between impulse and open direction
        langle_jaw = compute_angle_between(ldirection_jaw, limpulse_jaw)
        langle_hand = compute_angle_between(-ldirection_hand, limpulse_hand)

        lflag_jaw = (np.linalg.norm(limpulse_jaw) >= min_impulse) and np.rad2deg(
            langle_jaw
        ) <= max_angle
        lflag_hand = (np.linalg.norm(limpulse_hand) >= min_impulse) and np.rad2deg(
            langle_hand
        ) <= max_angle

        # -------------------
        rimpulse_jaw = get_pairwise_contact_impulse(
            contacts, self.right_jaw_link, actor
        )
        rimpulse_hand = get_pairwise_contact_impulse(
            contacts, self.right_hand_pitch_link, actor
        )

        # direction to open the gripper
        rdirection_jaw = self.right_jaw_link.pose.to_transformation_matrix()[:3, 1]
        rdirection_hand = self.right_hand_pitch_link.pose.to_transformation_matrix()[:3, 1]

        # angle between impulse and open direction
        rangle_jaw = compute_angle_between(rdirection_jaw, rimpulse_jaw)
        rangle_hand = compute_angle_between(-rdirection_hand, rimpulse_hand)

        rflag_jaw = (np.linalg.norm(rimpulse_jaw) >= min_impulse) and np.rad2deg(
            rangle_jaw
        ) <= max_angle
        rflag_hand = (np.linalg.norm(rimpulse_hand) >= min_impulse) and np.rad2deg(
            rangle_hand
        ) <= max_angle

        return all([lflag_jaw, lflag_hand, rflag_jaw, rflag_hand])

    def check_contact_left_jaw(self, actor: sapien.ActorBase, min_impulse=1e-6):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        limpulse_jaw = get_pairwise_contact_impulse(
            contacts, self.left_jaw_link, actor
        )
        limpulse_hand = get_pairwise_contact_impulse(
            contacts, self.left_hand_pitch_link, actor
        )

        return (
            np.linalg.norm(limpulse_jaw) >= min_impulse,
            np.linalg.norm(limpulse_hand) >= min_impulse,
        )
    
    def check_contact_right_jaw(self, actor: sapien.ActorBase, min_impulse=1e-6):
        assert isinstance(actor, sapien.ActorBase), type(actor)
        contacts = self.scene.get_contacts()

        rimpulse_jaw = get_pairwise_contact_impulse(
            contacts, self.right_jaw_link, actor
        )
        rimpulse_hand = get_pairwise_contact_impulse(
            contacts, self.right_hand_pitch_link, actor
        )

        return (
            np.linalg.norm(rimpulse_jaw) >= min_impulse,
            np.linalg.norm(rimpulse_hand) >= min_impulse,
        )

    @staticmethod
    def build_grasp_pose(approaching, closing, center):
        """
            Build a grasp pose (WidowX gripper).
            From link_gripper's frame, x=approaching, -y=closing
        """
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(approaching, closing)
        T = np.eye(4)
        T[:3, :3] = np.stack([approaching, closing, ortho], axis=1)
        T[:3, 3] = center
        return Pose.from_transformation_matrix(T)

    @property
    def base_pose(self):
        return self.base_link.get_pose()


# class WidowXBridgeDatasetCameraSetup(GrxRobot):
#     _config: defaults.WidowXBridgeDatasetCameraSetupConfig

#     @classmethod
#     def get_default_config(cls):
#         return defaults.WidowXBridgeDatasetCameraSetupConfig()


# class WidowXSinkCameraSetup(GrxRobot):
#     _config: defaults.WidowXSinkCameraSetupConfig

#     @classmethod
#     def get_default_config(cls):
#         return defaults.WidowXSinkCameraSetupConfig()

if __name__=="__main__":
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    grx = GrxRobot(scene= scene ,control_freq=60)
    grx._after_init()
    print(grx.get_fingers_info())
    print(grx.get_gripper_closedness())
    pass