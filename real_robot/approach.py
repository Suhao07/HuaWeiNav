"""Explicit standoff goal policy and lidar/base geometry for real navigation."""
import math
from real_robot.contracts import MotionGoal, MotionGoalMode, Pose3D


def approach_candidate(robot, yaw, center, standoff):
    """Produce a front-half-plane standoff candidate, never an object centroid goal."""
    if standoff <= 0 or not all(math.isfinite(v) for v in (*robot, yaw, *center, standoff)):
        raise ValueError('invalid approach geometry')
    dx,dy=center[0]-robot[0],center[1]-robot[1]
    distance=math.hypot(dx,dy)
    if dx*math.cos(yaw)+dy*math.sin(yaw)<=0:
        raise ValueError('selected object is not in front')
    if distance<=standoff+.15:
        raise ValueError('already within standoff')
    return [center[0]-standoff*dx/distance,center[1]-standoff*dy/distance,robot[2]]


def base_pose(position, quaternion, extrinsic):
    """Compose world-from-lidar with calibrated lidar-from-base XY/yaw transform."""
    x,y,z,w=quaternion;tx,ty,angle=extrinsic
    offset=[(1-2*(y*y+z*z))*tx+2*(x*y-z*w)*ty,
            2*(x*y+z*w)*tx+(1-2*(x*x+z*z))*ty,
            2*(x*z-y*w)*tx+2*(y*z+x*w)*ty]
    a,b=math.sin(angle/2),math.cos(angle/2)
    return [position[k]+offset[k] for k in range(3)],[x*b+y*a,y*b-x*a,z*b+w*a,w*b-z*a]


class StandoffGoalResolver:
    """Opt-in local approach policy; native planner still decides collision feasibility.

    Freeze a candidate per live UID to avoid continuously moving the action goal.
    No historical identity binding or reached/STOP evidence is produced here.
    """
    def __init__(self, standoff=.8, xy_tolerance=.1):
        if not math.isfinite(standoff) or standoff<=0:
            raise ValueError('standoff must be finite and positive')
        if not math.isfinite(xy_tolerance) or xy_tolerance<=0:
            raise ValueError("xy_tolerance must be finite and positive")
        self.xy_tolerance=xy_tolerance
        self.standoff=standoff
        self.goals={}

    def resolve(self, intent, snapshot, excluded_viewpoint_ids):
        """Resolve a selected live object; return None for unsupported/unsafe candidates."""
        if intent.mode in {MotionGoalMode.WAIT,MotionGoalMode.STOP}:
            return intent.to_motion_goal()
        if intent.mode != MotionGoalMode.GO_TO_OBJECT:
            return None
        uid=intent.target_object_uid;obj=snapshot.object_by_uid(uid)
        key='standoff:'+str(uid)
        if obj is None or obj.position is None or key in excluded_viewpoint_ids:
            return None
        if uid in self.goals:
            return self.goals[uid]
        robot=snapshot.robot_pose;q=robot.orientation_xyzw
        yaw=math.atan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))
        try:
            position=approach_candidate(robot.position,yaw,obj.position,self.standoff)
        except ValueError:
            return None
        if math.dist(position[:2],robot.position[:2]) <= self.xy_tolerance:
            return None
        goal=MotionGoal(tolerance={"xy_goal_tolerance_m":self.xy_tolerance},mode=intent.mode,goal_pose=Pose3D(tuple(position),robot.orientation_xyzw,robot.frame_id,snapshot.timestamp),
            target_object_uid=uid,reason=intent.reason,metadata={**intent.metadata,
            'viewpoint_uid':key,'viewpoint_source':'local_standoff_candidate',
            'standoff_m':self.standoff,'collision_checked_by':'native_local_planner',
            'look_at_required':False})
        self.goals[uid]=goal
        return goal
