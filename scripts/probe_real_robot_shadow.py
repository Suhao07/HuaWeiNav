#!/usr/bin/env python3
"""Bounded live chair planning acceptance with isolated, non-actuating outputs.

Uses the current runtime-selected identity; approach geometry is a shadow-only
candidate, not a verified historical viewpoint or permission to drive.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path


from real_robot.approach import approach_candidate, base_pose


def main():
    """Observe live input, publish one isolated candidate, and persist evidence."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from nav_msgs.msg import Odometry, Path as NavPath
    from geometry_msgs.msg import PointStamped, TwistStamped
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String
    from tare_planner.msg import ObjectNodeList
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--standoff', type=float, default=.8)
    parser.add_argument('--duration', type=float, default=35.)
    parser.add_argument('--odom', default='/aft_mapped_to_init')
    parser.add_argument('--cloud', default='/cloud_registered')
    parser.add_argument('--base-in-lidar', type=float, nargs=3, default=[0.,0.,0.], metavar=('X','Y','YAW'))
    parser.add_argument('--objects', default='/huawei_vln/d435i_object_nodes_list')
    args = parser.parse_args()
    rclpy.init(); node = rclpy.create_node('strive_shadow_acceptance')
    result = dict(instruction='前方 chair', scope='live sensors, shadow approach and native planner/follower; bridge off',
                  paths=[], velocities=[], raw_velocities=[], statuses=[], counts=dict(odom=0,cloud=0,objects=0), success=False)
    latest = {}; received = {}; start = time.monotonic(); goal_time = None
    base_pub=node.create_publisher(Odometry,'/strive/shadow/base_odom',10)
    def store(key, msg):
        if key == 'odom':
            msg=copy.deepcopy(msg);p=msg.pose.pose.position;q=msg.pose.pose.orientation
            xyz,quat=base_pose([p.x,p.y,p.z],[q.x,q.y,q.z,q.w],args.base_in_lidar)
            p.x,p.y,p.z=xyz;q.x,q.y,q.z,q.w=quat
            msg.child_frame_id='shadow_base'
            # Pose-only planner input; no claim of transformed twist/covariance.
            msg.twist.twist=type(msg.twist.twist)()
            msg.pose.covariance=[0.]*36;msg.twist.covariance=[0.]*36
            base_pub.publish(msg)
        latest[key]=msg; received[key]=time.monotonic(); result['counts'][key]+=1
    subscriptions=[]
    for typ,topic,key in [(Odometry,args.odom,'odom'),(PointCloud2,args.cloud,'cloud'),(ObjectNodeList,args.objects,'objects')]:
        subscriptions.append(node.create_subscription(typ,topic,lambda m,k=key:store(k,m),qos_profile_sensor_data))
    def path(m):
        if goal_time is not None and len(m.poses)>1:
            result['paths'].append(dict(frame=m.header.frame_id,elapsed=time.monotonic()-start,
                points=[[p.pose.position.x,p.pose.position.y,p.pose.position.z] for p in m.poses]))
    def velocity(m):
        if goal_time is not None:
            result['velocities'].append([time.monotonic()-start,m.twist.linear.x,m.twist.linear.y,m.twist.angular.z])
    subscriptions.append(node.create_subscription(TwistStamped,'/strive/shadow/raw_cmd_vel',
        lambda m:result['raw_velocities'].append([time.monotonic()-start,m.twist.linear.x,m.twist.linear.y,m.twist.angular.z]),10))
    subscriptions.extend([node.create_subscription(NavPath,'/strive/shadow/path',path,10),
        node.create_subscription(TwistStamped,'/strive/shadow/cmd_vel',velocity,10),
        node.create_subscription(String,'/strive/shadow/status',lambda m:result['statuses'].append(m.data),10)])
    pub=node.create_publisher(PointStamped,'/strive/shadow/way_point',10)
    try:
        while time.monotonic()-start < args.duration:
            rclpy.spin_once(node,timeout_sec=.05)
            names=node.get_node_names()
            if any('agilex_platform' in n for n in names):
                raise RuntimeError('chassis bridge present; refusing shadow acceptance')
            endpoints=node.get_subscriptions_info_by_topic('/strive/shadow/cmd_vel')
            if any(e.node_name != node.get_name() for e in endpoints):
                raise RuntimeError('unexpected shadow velocity subscriber')
            if goal_time is None and time.monotonic()-start>3 and len(latest)==3:
                if any(time.monotonic()-received[k]>3 for k in latest): continue
                lines=Path(args.decisions).read_text().splitlines()
                decision=next((json.loads(line) for line in reversed(lines) if json.loads(line).get('intent',{}).get('target_object_uid')),None)
                if not decision or abs(time.time()-decision['timestamp'])>5: continue
                uid=decision['intent']['target_object_uid']
                objects=[o for o in latest['objects'].nodes if o.object_id and uid=='sysnav_object:'+str(o.object_id[0]) and 'chair' in o.label.lower()]
                if not objects: continue
                odom=latest['odom']; obj=objects[0]; p=odom.pose.pose.position; q=odom.pose.pose.orientation
                yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
                robot=[p.x,p.y,p.z]; center=[obj.position.x,obj.position.y,obj.position.z]
                if latest['cloud'].header.frame_id != odom.header.frame_id or latest['objects'].header.frame_id != odom.header.frame_id:
                    raise RuntimeError('input world frames differ')
                goal=approach_candidate(robot,yaw,center,args.standoff)
                msg=PointStamped();msg.header=odom.header;msg.point.x,msg.point.y,msg.point.z=goal
                result.update(target_uid=uid,target_label=obj.label,target_center=center,robot=robot,yaw=yaw,
                    goal=goal,standoff_m=args.standoff,base_in_lidar=args.base_in_lidar,world_frame=odom.header.frame_id,
                    target_selection_source='current production runtime decision; front geometry checked',
                    candidate_source='shadow standoff, not production viewpoint resolver',decision_timestamp=decision['timestamp'])
                pub.publish(msg);goal_time=time.monotonic()
        result['nodes']=node.get_node_names()
        result['physical_publishers']={t:node.count_publishers(t) for t in ['/cmd_vel','/cmd_vel/autonomy','/navflow_cmd_vel','/way_point']}
        result['nonzero_velocity_count']=sum(math.hypot(v[1],v[2])>.001 or abs(v[3])>.001 for v in result['velocities'])
        result['success']=bool(result['paths'] and result['nonzero_velocity_count'] and not any(result['physical_publishers'].values()))
    except Exception as error:
        result['error']=str(error)
    finally:
        result['duration_s']=time.monotonic()-start
        dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('paths','velocities','raw_velocities','statuses')}))
        node.destroy_node();rclpy.shutdown()
    return 0 if result['success'] else 1

if __name__=='__main__':
    raise SystemExit(main())
