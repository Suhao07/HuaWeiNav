"""Production standoff resolution and default bridge isolation regressions."""
import os
import subprocess
from pathlib import Path
from real_robot.approach import StandoffGoalResolver
from real_robot.contracts import NavigationIntent, MotionGoalMode, Pose3D, SemanticMapSnapshot, ObjectNodeSnapshot

ROOT=Path(__file__).resolve().parents[1]

def snap(x=2):
    return SemanticMapSnapshot(timestamp=10.,robot_pose=Pose3D((0,0,0)),
        objects=(ObjectNodeSnapshot(uid='live',label='chair',position=(x,0,0)),))


def test_production_goal_is_stable_and_not_centroid():
    resolver=StandoffGoalResolver(.8)
    intent=NavigationIntent(mode=MotionGoalMode.GO_TO_OBJECT,target_object_uid='live')
    goal=resolver.resolve(intent,snap(),set())
    assert goal.goal_pose.position==(1.2,0,0)
    assert resolver.resolve(intent,snap(2.2),set()) is goal
    assert resolver.resolve(intent,snap(),{'standoff:live'}) is None
    assert goal.metadata['viewpoint_source']=='local_standoff_candidate'


def test_behind_and_close_targets_do_not_generate_motion():
    intent=NavigationIntent(mode=MotionGoalMode.GO_TO_OBJECT,target_object_uid='live')
    for x in [-1,.5]:
        assert StandoffGoalResolver().resolve(intent,snap(x),set()) is None


def test_preview_default_does_not_require_or_enable_bridge():
    env={**os.environ,'STRIVE_EXECUTION_MODE':'preview','ALLOW_REAL_MOTION':'true'}
    run=subprocess.run(['bash',str(ROOT/'scripts/start_real_robot_navigation.sh'),'--check'],env=env,capture_output=True,text=True)
    assert run.returncode==0
    assert 'bridge_started=false' in run.stdout and '/strive/preview/cmd_vel' in run.stdout


def test_live_requires_separate_motion_authorization_and_contract():
    for motion,contract in [('false',''),('true','/does/not/exist')]:
        env={**os.environ,'STRIVE_EXECUTION_MODE':'live','ALLOW_REAL_MOTION':motion,'CONTROL_CONTRACT_FILE':contract}
        run=subprocess.run(['bash',str(ROOT/'scripts/start_real_robot_navigation.sh'),'--check'],env=env,capture_output=True,text=True)
        assert run.returncode!=0


def test_live_readiness_waits_for_fresh_bridge_authorization():
    import ast
    from types import SimpleNamespace as NS
    from real_robot.sysnav_runtime import RuntimeReadiness
    path=ROOT/'real_robot/ros2_ws/src/strive_sysnav_bringup/strive_sysnav_bringup/instruction_runtime_node.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='StriveInstructionRuntimeNode')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_readiness')
    env={'RuntimeReadiness':RuntimeReadiness}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),env)
    node=NS(require_motion_ready=True,_motion_ready=False,_motion_ready_stamp=10.,_now_seconds=lambda:10.1,
        semantic_bridge=NS(has_object_snapshot=lambda:True),require_pose=True,_latest_pose=object(),
        require_image=True,_latest_image_stamp=10.,dry_run=False,_runtime_safety_metadata=lambda:{})
    assert not env['_readiness'](node).ready
    node._motion_ready=True
    assert env['_readiness'](node).ready
    node._motion_ready_stamp=9.
    assert not env['_readiness'](node).ready


def test_action_terminal_clears_planner_before_reporting_result():
    import ast
    import threading
    from types import SimpleNamespace as NS
    from typing import Any
    from real_robot.contracts import NavigationStatusCode
    path=ROOT/'real_robot/ros2_ws/src/strive_sysnav_motion/strive_sysnav_motion/motion_server.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SysNavMotionServer')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_execute_callback')
    env={'Any':Any,'rclpy':NS(ok=lambda:True),'NavigationStatusCode':NavigationStatusCode,
         'motion_goal_from_action_goal':lambda *a:NS(look_at=None)}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),env)
    for code in (NavigationStatusCode.BLOCKED,NavigationStatusCode.TIMEOUT,NavigationStatusCode.REACHED):
        calls=[];status=NS(status=code,is_terminal=lambda:True)
        node=NS(_active_goal_lock=threading.Lock(),_active_goal_handle=None,world_frame='map',
            _normalize_goal_request=lambda x:x,
            controller=NS(send_goal=lambda x:'goal',poll_status=lambda x:status,cancel=lambda x:calls.append(('cancel',x))),
            _publish_feedback=lambda *a:None,
            _finish_from_status=lambda handle,s:(calls.append(('report',s.status)),s)[1])
        result=env['_execute_callback'](node,NS(request=object(),is_cancel_requested=False))
        assert result is status
        assert calls==[('cancel','goal'),('report',code)]
        assert node._active_goal_handle is None


def test_standoff_tolerance_survives_action_serialization(monkeypatch):
    from types import SimpleNamespace as NS
    from real_robot.action_motion_controller import RosActionMotionController
    from real_robot.ros_motion_action import motion_goal_from_action_goal
    intent=NavigationIntent(mode=MotionGoalMode.GO_TO_OBJECT,target_object_uid='live')
    goal=StandoffGoalResolver(.8,.1).resolve(intent,snap(1.13),set())
    assert goal is not None and goal.tolerance['xy_goal_tolerance_m']==.1
    # The real action message builder requires no ROS middleware with a supplied message type.
    def message():
        return NS(target_pose=NS(header=NS(frame_id='',stamp=None),pose=NS(position=NS(),orientation=NS())),
                  look_at=NS(header=NS(),point=NS()))
    controller=object.__new__(RosActionMotionController)
    controller._action_type=NS(Goal=message)
    monkeypatch.setattr('real_robot.action_motion_controller._pose_stamped',lambda p:
        NS(header=NS(frame_id=p.frame_id,stamp=NS(sec=10,nanosec=0)),
           pose=NS(position=NS(x=p.position[0],y=p.position[1],z=p.position[2]),
                   orientation=NS(x=0.,y=0.,z=0.,w=1.))))
    request=controller._goal_message(goal)
    assert request.xy_tolerance_m==.1
    restored=motion_goal_from_action_goal(request,'map')
    assert restored.tolerance['xy_goal_tolerance_m']==.1
    assert StandoffGoalResolver(.8,.35).resolve(intent,snap(1.13),set()) is None
