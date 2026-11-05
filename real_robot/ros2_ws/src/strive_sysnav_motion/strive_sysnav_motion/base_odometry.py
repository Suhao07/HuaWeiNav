"""Convert lidar odometry to the calibrated base pose in the same world frame."""
import copy
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from real_robot.approach import base_pose


class BaseOdometry(Node):
    """Publish base odometry without resetting the saved-map world origin."""
    def __init__(self):
        super().__init__('strive_base_odometry')
        for key,value in dict(input_topic='/aft_mapped_to_init',output_topic='/strive/base_odom',
                              world_frame='camera_init',base_frame='base_link',base_x=-.2,base_y=0.,base_yaw=-1.5708).items():
            self.declare_parameter(key,value)
        self.extrinsic=[float(self.get_parameter(k).value) for k in ('base_x','base_y','base_yaw')]
        if not all(math.isfinite(v) for v in self.extrinsic): raise ValueError('invalid base extrinsic')
        tx,ty,a=self.extrinsic;c,s=math.cos(a),math.sin(a)
        rotation=np.array([[c,s,0],[-s,c,0],[0,0,1.]])
        skew=np.array([[0,0,ty],[0,0,-tx],[-ty,tx,0]])
        self.adjoint=np.block([[rotation,-rotation@skew],[np.zeros((3,3)),rotation]])
        self.publisher=self.create_publisher(Odometry,str(self.get_parameter('output_topic').value),10)
        self.create_subscription(Odometry,str(self.get_parameter('input_topic').value),self.update,qos_profile_sensor_data)

    def update(self, msg):
        """Reject wrong-frame/nonfinite input; transform pose and child-frame twist."""
        if msg.header.frame_id!=self.get_parameter('world_frame').value:
            self.get_logger().error('Rejecting odometry in unexpected world frame');return
        p=msg.pose.pose.position;q=msg.pose.pose.orientation
        values=[p.x,p.y,p.z,q.x,q.y,q.z,q.w]
        norm=math.sqrt(sum(v*v for v in values[3:]))
        if not all(math.isfinite(v) for v in values) or abs(norm-1)>.01: return
        out=copy.deepcopy(msg)
        xyz,quat=base_pose(values[:3],[v/norm for v in values[3:]],self.extrinsic)
        out.pose.pose.position.x,out.pose.pose.position.y,out.pose.pose.position.z=xyz
        out.pose.pose.orientation.x,out.pose.pose.orientation.y,out.pose.pose.orientation.z,out.pose.pose.orientation.w=quat
        out.child_frame_id=str(self.get_parameter('base_frame').value)
        v=msg.twist.twist.linear;w=msg.twist.twist.angular
        twist=self.adjoint@np.array([v.x,v.y,v.z,w.x,w.y,w.z])
        if not np.isfinite(twist).all(): return
        out.twist.twist.linear.x,out.twist.twist.linear.y,out.twist.twist.linear.z=twist[:3].tolist()
        out.twist.twist.angular.x,out.twist.twist.angular.y,out.twist.twist.angular.z=twist[3:].tolist()
        out.twist.covariance=(self.adjoint@np.array(msg.twist.covariance).reshape(6,6)@self.adjoint.T).ravel().tolist()
        # Pose covariance is not fused here; consumers use the source localization.
        self.publisher.publish(out)


def main(args=None):
    """Run the base odometry adapter until shutdown."""
    rclpy.init(args=args);node=BaseOdometry()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()


if __name__=='__main__':main()
