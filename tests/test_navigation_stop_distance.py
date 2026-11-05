"""Offline native-follower counterexample and formal-entry parameter regressions."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace as NS

import pytest

from real_robot.navigation_parameters import DEFAULTS, validate_parameters
from real_robot.contracts import MotionGoal, MotionGoalMode, NavigationStatusCode, Pose3D
from real_robot.sysnav_ros_adapters import RosNavigationStatusProvider

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'real_robot/ros2_ws/src/local_planner/src/pathFollower.cpp'


@pytest.fixture(scope='module')
def native_trace(tmp_path_factory):
    # Compile the unchanged production speed-selection/ramp block, not a Python
    # reimplementation. ROS, obstacles and dynamics are outside this mechanism test.
    source = NATIVE.read_text()
    block = source.split('      float joySpeed2 = maxSpeed * joySpeed;', 1)[1]
    block = 'float joySpeed2 = maxSpeed * joySpeed;' + block.split('      if (odomTime < stopInitTime', 1)[0]
    defaults = []
    for name in ['yawRateGain', 'stopYawRateGain', 'dirDiffThre', 'omniDirGoalThre',
                 'omniDirDiffThre', 'slowRate1', 'slowRate2', 'slowRate3', 'slowTime1', 'slowTime2']:
        defaults.append(re.search(r'double ' + name + r' = [^;]+;', source).group())
    code = '''#include <cmath>
#include <iostream>
#include <cstdlib>
using namespace std;
int main(int argc, char** argv) {
 double stopDisThre=atof(argv[1]), remaining=atof(argv[2]);
 const double maxSpeed=.15, maxAccel=.15, maxYawRate=20, PI=acos(-1), slowDwnDisThre=1.;
 float joySpeed=1, joyYaw=0, vehicleSpeed=0, vehicleYawRate=0, dirDiff=0;
 bool navFwd=true, autonomyMode=true, noRotAtGoal=true;
 double slowInitTime=0; int slowDown=0;
''' + '\n'.join(defaults) + '''
 for(int i=0; i<3000; ++i) {
   double odomTime=i*.01;
   // Ideal straight replanning, endpoint cropped to a 1 cm grid and a 0.75 m horizon.
   float dis=fmin(.75, floor((remaining+1e-9)/.01)*.01), endDis=dis;
   int pathSize=static_cast<int>(dis/.01)+1;
''' + block + '''
   cout << odomTime << " " << remaining << " " << vehicleSpeed << "\\n";
   remaining -= vehicleSpeed*.01;
 }
}
'''
    folder = tmp_path_factory.mktemp('native-stop')
    cpp = folder / 'probe.cpp'; exe = folder / 'probe'
    cpp.write_text(code)
    subprocess.run(['g++', '-std=c++17', '-O2', str(cpp), '-o', str(exe)], check=True, capture_output=True)

    def trace(stop, distance):
        run = subprocess.run([str(exe), str(stop), str(distance)], check=True, capture_output=True, text=True)
        return [tuple(map(float, line.split())) for line in run.stdout.splitlines()]
    return trace


def assess(trace, distance, tolerance=.1):
    clock = {'t': 0.0}
    provider = RosNavigationStatusProvider(xy_tolerance_m=tolerance, stable_reach_time_s=.2,
        velocity_tolerance_mps=.08, now_fn=lambda: clock['t'])
    goal = MotionGoal(MotionGoalMode.GO_TO_OBJECT, goal_pose=Pose3D((distance, 0, 0)),
                      tolerance={'xy_goal_tolerance_m': tolerance})
    for time, remaining, velocity in trace:
        clock['t'] = time
        provider.latest_pose = Pose3D((distance-remaining, 0, 0))
        provider.latest_velocity = (velocity, 0, 0)
        status = provider('offline-stop', goal)
        if status.is_terminal():
            return {'status': status.status.value, 'time_s': time,
                    'remaining_m': remaining, 'velocity_mps': velocity}
    pytest.fail('No terminal status within synthetic trace')


@pytest.mark.parametrize('distance', [.15, .34, .45])
def test_old_stop_blocks_outside_tolerance_new_stop_enters_it(native_trace, distance):
    old = assess(native_trace(.20, distance), distance)
    new = assess(native_trace(.05, distance), distance)
    assert old['status'] == NavigationStatusCode.BLOCKED.value and old['remaining_m'] > .1
    assert new['status'] == NavigationStatusCode.REACHED.value and new['remaining_m'] <= .1
    print(json.dumps({'initial_distance_m': distance, 'old': old, 'new': new}, sort_keys=True))


def test_already_inside_tolerance_and_fixed_pose_no_false_progress(native_trace):
    assert assess(native_trace(.05, .08), .08)['status'] == NavigationStatusCode.REACHED.value
    fixed = [(i*.01, .34, 0.) for i in range(3000)]
    assert assess(fixed, .34)['status'] == NavigationStatusCode.BLOCKED.value


@pytest.mark.parametrize('name', list(DEFAULTS))
@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'invalid'])
def test_invalid_distance_rejected(name, value):
    with pytest.raises(ValueError):
        validate_parameters({name: value})


@pytest.mark.parametrize('values', [
    {'NAV_FOLLOWER_STOP_DISTANCE_M': '1'},
    {'NAV_FOLLOWER_STOP_DISTANCE_M': '2'},
    {'NAV_GOAL_TOLERANCE_M': '.04'},
    {'NAV_FOLLOWER_LOOKAHEAD_M': '.5'},
])
def test_incompatible_thresholds_rejected(values):
    with pytest.raises(ValueError):
        validate_parameters(values)


def test_host_and_container_check_custom_parameters_without_starting_ros(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('NAV_')}
    env.update(STRIVE_EXECUTION_MODE='preview', NAV_FOLLOWER_STOP_DISTANCE_M='.03',
               NAV_FOLLOWER_LOOKAHEAD_M='.4', NAV_FOLLOWER_SLOWDOWN_DISTANCE_M='.8')
    for script in ['run_real_robot_navigation.sh', 'start_real_robot_navigation.sh']:
        command = ['bash', str(ROOT/'scripts'/script), '--check']
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        assert 'bridge_started=false' in run.stdout
        for name in DEFAULTS:
            assert f'{name}={float(env.get(name, DEFAULTS[name])):g}' in run.stdout
        bad = subprocess.run(command, env={**env, 'NAV_FOLLOWER_STOP_DISTANCE_M': '2'}, capture_output=True, text=True)
        assert bad.returncode != 0 and 'must be smaller' in bad.stderr


def test_launch_wiring_and_docker_forwarding():
    # Evaluate the actual launch function with small middleware-free constructors.
    path = ROOT/'real_robot/ros2_ws/src/strive_sysnav_motion/launch/sysnav_lower_stack.launch.py'
    fn = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef))
    defaults = {}
    def declare(name, default_value):
        defaults[name] = default_value
    env = dict(DeclareLaunchArgument=declare, LaunchConfiguration=lambda n: defaults[n],
        LaunchDescription=lambda items: items, Node=lambda **kw: NS(**kw),
        IfCondition=lambda x: x, FindPackageShare=lambda x: x, PathJoinSubstitution=lambda x: x)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
    nodes = env['generate_launch_description']()
    follower = next(n for n in nodes if getattr(n, 'executable', None) == 'pathFollower')
    params = {k: v for entry in follower.parameters for k, v in entry.items()}
    mapping = {'NAV_FOLLOWER_STOP_DISTANCE_M': ('stopDisThre', 'follower_stop_distance_m'),
               'NAV_FOLLOWER_LOOKAHEAD_M': ('lookAheadDis', 'follower_lookahead_m'),
               'NAV_FOLLOWER_SLOWDOWN_DISTANCE_M': ('slowDwnDisThre', 'follower_slowdown_distance_m')}
    script = (ROOT/'scripts/start_real_robot_navigation.sh').read_text()
    whitelist = (ROOT/'docker/run_real_robot_sysnav_stack.sh').read_text().split('RUNTIME_ENVS=()', 1)[1].split('do', 1)[0]
    for name, (native, launch) in mapping.items():
        assert float(params[native]) == DEFAULTS[name]
        fallback = re.search(re.escape(launch) + r':="\$\{' + name + r':-([^}]+)\}"', script)
        assert fallback and float(fallback[1]) == DEFAULTS[name]
        assert name in whitelist


@pytest.mark.parametrize('distance', [1.2, 1.6, 2.0])
def test_requested_half_meter_stop_one_meter_arrival(native_trace, distance):
    result = assess(native_trace(DEFAULTS['NAV_FOLLOWER_STOP_DISTANCE_M'], distance),
                    distance, DEFAULTS['NAV_GOAL_TOLERANCE_M'])
    assert result['status'] == NavigationStatusCode.REACHED.value
    assert result['remaining_m'] <= 1.0
    assert abs(result['velocity_mps']) <= .08
    print(json.dumps({'requested_defaults_initial_m': distance, 'result': result}, sort_keys=True))


def test_requested_tolerance_does_not_force_nearby_candidate():
    from real_robot.approach import StandoffGoalResolver
    from real_robot.contracts import NavigationIntent, ObjectNodeSnapshot, SemanticMapSnapshot
    intent = NavigationIntent(MotionGoalMode.GO_TO_OBJECT, target_object_uid='chair')
    def snapshot(center):
        return SemanticMapSnapshot(timestamp=0., robot_pose=Pose3D((0,0,0)),
            objects=(ObjectNodeSnapshot(uid='chair', label='chair', position=(center,0,0)),))
    resolver = StandoffGoalResolver(.8, DEFAULTS['NAV_GOAL_TOLERANCE_M'])
    assert resolver.resolve(intent, snapshot(1.14), set()) is None
    goal = resolver.resolve(intent, snapshot(2.), set())
    assert goal is not None and goal.tolerance['xy_goal_tolerance_m'] == 1.0
    fixed = [(i*.01, 1.2, 0.) for i in range(3000)]
    assert assess(fixed, 1.2, 1.)['status'] == NavigationStatusCode.BLOCKED.value
