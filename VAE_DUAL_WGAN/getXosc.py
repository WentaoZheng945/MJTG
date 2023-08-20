# encoding=utf-8
# Author：Wentao Zheng
# E-mail: swjtu_zwt@163.com
# developed time: 2023/7/17 15:53
import math
import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET


class Repair_cutin():
    def __init__(self, frame_rate=0.04, lon_bound=[-8, 5], lat_bound=[-3.5, 3.5], fusion_path='', isNorm=True, isXy=True, lims=[]) -> None:
        self.rate = frame_rate
        self.isNorm = isNorm
        self.isXy = isXy
        self.lims = lims
        self.lon_bound = lon_bound
        self.lat_bound = lat_bound
        self.fusion = np.load(fusion_path, allow_pickle=True)
        self.fusion_re = np.array([])
        self.sceneRecoverPosition()
        print('over')

    def restore(self, pos, min_p, max_p):
        '''将归一化的参数还原
        :param pos        : 某一维参数序列对应的列表
        :param min_p      : 该参数归一化时对应的下界
        :param max_p      : 该参数归一化对应的上界

        :return pos       : 完成还原的参数序列
        '''
        pos = [min_p + i * (max_p - min_p) for i in pos]
        return pos

    def sceneRecoverPosition(self):
        '''将传入的场景参数还原
        1. 将归一化的车辆坐标参数fusion，还原为原始量纲fusion_re
        2. 基于还原的fusion_re=fusion_xy，计算车辆速度加速度
        '''
        print(self.fusion.shape[0])
        for i in tqdm(range(self.fusion.shape[0])):
            x1 = self.fusion[i, :, 0].tolist()
            y1 = self.fusion[i, :, 1].tolist()
            x2 = self.fusion[i, :, 2].tolist()
            y2 = self.fusion[i, :, 3].tolist()
            x3 = self.fusion[i, :, 4].tolist()
            y3 = self.fusion[i, :, 5].tolist()

            # TODO: 将归一化的数据还原，并储存在fusion_re中
            if self.isNorm:
                x1 = self.restore(x1, self.lims[0], self.lims[1])
                y1 = self.restore(y1, self.lims[2], self.lims[3])
                x2 = self.restore(x2, self.lims[4], self.lims[5])
                y2 = self.restore(y2, self.lims[6], self.lims[7])
                x3 = self.restore(x3, self.lims[8], self.lims[9])
                y3 = self.restore(y3, self.lims[10], self.lims[11])
            temp_array1 = np.empty((len(x1), 6), dtype=np.float64)
            temp_array1[:, 0] = x1
            temp_array1[:, 1] = y1
            temp_array1[:, 2] = x2
            temp_array1[:, 3] = y2
            temp_array1[:, 4] = x3
            temp_array1[:, 5] = y3
            if self.fusion_re.size > 0:
                self.fusion_re = np.concatenate((self.fusion_re, temp_array1.reshape([1, -1, 6])), axis=0)
            else:
                self.fusion_re = temp_array1.reshape([1, -1, 6])
        return


# TODO：进行TREE的美化，即对根节点下的每个子节点进行相应的换行与缩进操作
def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    '''
    该部分的美化操作，置于整个tree构建完成后，进行统一美化
    :param arguments: 传进来的Elment类，缩进参数，换行参数
    :return: None

    pretty_xml(root, '\t', '\n')  # 执行美化方法
    tree.write('.xdor')  # 美化完成后将XML写入.xdor
    '''
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素，说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作


def xosc_write(data, xodr_path, output_path, flag=0, is_exam=False):
    '''
    该方法用于自动化实现场景的csv文件到openScenario格式的自动转换
    转换思路：将每辆车的轨迹按照轨迹逐帧记录storyBoard中的Event中

    Input：轨迹csv文本路径、OpenDrive文件路径、输出的OpenScenario文件路径、相对于OpenDive的y坐标偏移量
    Output：None
    '''
    root = ET.Element('OpenSCENARIO')
    # root下第一层目录构建——Level 1
    header = ET.SubElement(root, 'FileHeader')
    header.attrib = {'revMajor': '1', 'revMinor': '0', 'date': '2021-11-02T16:20:00', 'description': 'scenario_highD', 'author': 'OnSite_TOPS'}
    pareDecl = ET.SubElement(root, 'ParameterDeclarations')
    pareDecl.attrib = {}
    catalog = ET.SubElement(root, 'CatalogLocations')
    catalog.attrib = {}
    roadNet = ET.SubElement(root, 'RoadNetwork')
    roadNet.attrib = {}
    entity = ET.SubElement(root, 'Entities')
    entity.attrib = {}
    storyboard = ET.SubElement(root, 'Storyboard')
    storyboard.attrib = {}

    # root下第二层目录构建——Level 2
    logic = ET.SubElement(roadNet, 'LogicFile')
    logic.attrib = {'filepath': xodr_path}
    init = ET.SubElement(storyboard, 'Init')
    init.attrib = {}
    story = ET.SubElement(storyboard, 'Story')
    story.attrib = {'name': 'Cutin'}
    stoptrigger = ET.SubElement(storyboard, 'StopTrigger')
    stoptrigger.attrib = {}

    # root下第三层目录构建——Level 3
    actions = ET.SubElement(init, 'Actions')
    actions.attrib = {}
    paramDecl = ET.SubElement(story, 'ParameterDeclarations')
    paramDecl.attrib = {}

    # Level 4及以下，以模块为单位进行树结构构建
    '''Init-Action下的GlobalAction块'''
    globalaction = ET.Element('GlobalAction', {})
    actions.append(globalaction)
    environmentaction = ET.Element('EnvironmentAction', {})
    globalaction.append(environmentaction)
    environment = ET.Element('Environment', {'name': 'Default_Environment'})
    environmentaction.append(environment)
    timeofday = ET.Element('TimeOfDay', {'animation': 'false', 'dateTime': '2021-12-10T11:00:00'})
    environment.append(timeofday)
    weather = ET.Element('Weather', {'cloudState': 'free'})
    environment.append(weather)
    sun = ET.SubElement(weather, 'Sun')
    sun.attrib = {'intensity': '1.0', 'azimuth': '0.0', 'elevation': '1.571'}
    fog = ET.SubElement(weather, 'Fog')
    fog.attrib = {'visualRange': '100000.0'}
    precip = ET.SubElement(weather, 'Precipitation')
    precip.attrib = {'precipitationType': 'dry', 'intensity': '0.0'}
    roadcondi = ET.Element('RoadCondition', {'frictionScaleFactor': '1.0'})
    environment.append(roadcondi)
    ''''读取场景cvs后，自动化完成车辆初始化及轨迹设置'''
    # y_bias = 18.175  # highD坐标系与OpenDRIVE坐标系转换
    sample = 1 / 25  # highD场景的轨迹采样率（读取recordingMeta）
    for count in range(3):
        x_list = data[:, 2 * count]
        y_list = data[:, 2 * count + 1]
        if count == 0:  # ego
            whole_time = len(x_list) * sample  # 场景总时间
            # 申明Entities及对应属性
            scenObj = ET.SubElement(entity, 'ScenarioObject')
            scenObj.attrib = {'name': 'Ego'}
            veh = ET.SubElement(scenObj, 'Vehicle')
            if flag == 0:
                veh.attrib = {'name': 'Default_car', 'vehicleCategory': 'car'}
            else:
                veh.attrib = {'name': 'Default_car', 'vehicleCategory': 'car', 'model3d': 'car_white.osgb'}
            boundingbox = ET.SubElement(veh, 'BoundingBox')  # 车辆边框属性设置
            center = ET.SubElement(boundingbox, 'Center')  # 车辆中心在【车辆坐标系】中的坐标
            center.attrib = {'x': '%.16e' % 1.5, 'y': '%.16e' % 0, 'z': '%.16e' % 0.9}
            dimension = ET.SubElement(boundingbox, 'Dimensions')
            dimension.attrib = {'width': '%.16e' % 2.1, 'length': '%.16e' % 4.5, 'height': '%.16e' % 1.8}
            performance = ET.SubElement(veh, 'Performance')
            performance.attrib = {'maxSpeed': "200", 'maxAcceleration': "200", 'maxDeceleration': "10.0"}
            axles = ET.SubElement(veh, 'Axles')
            axles.attrib = {}
            front = ET.SubElement(axles, 'FrontAxle')
            front.attrib = {'maxSteering': "0.5", 'wheelDiameter': "0.5", 'trackWidth': "1.75", 'positionX': "2.8", 'positionZ': "0.25"}
            rear = ET.SubElement(axles, 'RearAxle')
            rear.attrib = {'maxSteering': "0.0", 'wheelDiameter': "0.5", 'trackWidth': "1.75", 'positionX': "0.0", 'positionZ': "0.25"}
            property = ET.SubElement(veh, 'Properties')
            property.attrib = {}
            controller = ET.SubElement(scenObj, 'ObjectController')
            controller.attrib = {}
            # Init部分对车辆的属性设置
            private_ego = ET.Element('Private', {'entityRef': 'Ego'})  # 申明专属对象
            actions.append(private_ego)
            if not is_exam:
                privateAction1_init = ET.SubElement(private_ego, 'PrivateAction')  # 初始化专属动作1（速度）
                privateAction1_init.attrib = {}
                longitAction = ET.SubElement(privateAction1_init, 'LongitudinalAction')
                longitAction.attrib = {}
                speedAction = ET.SubElement(longitAction, 'SpeedAction')
                speedAction.attrib = {}
                speedAcDy = ET.SubElement(speedAction, 'SpeedActionDynamics')
                speedAcDy.attrib = {'dynamicsShape': 'step', 'value': '0', 'dynamicsDimension': 'time'}
                speedAcTar = ET.SubElement(speedAction, 'SpeedActionTarget')
                speedAcTar.attrib = {}
                absTarSpeed = ET.SubElement(speedAcTar, 'AbsoluteTargetSpeed')
                v_0 = (x_list[2] - x_list[0]) / float(2 * sample)
                absTarSpeed.attrib = {'value': '%.16e' % v_0}

                privateAction2_init = ET.SubElement(private_ego, 'PrivateAction')  # 初始化专属动作2（位置）
                privateAction2_init.attrib = {}
                telepAction = ET.SubElement(privateAction2_init, 'TeleportAction')
                telepAction.attrib = {}
                position_init = ET.SubElement(telepAction, 'Position')
                position_init.attrib = {}
                worldPos_init = ET.SubElement(position_init, 'WorldPosition')  # 采用全局世界坐标对车辆进行定位
                heading = math.atan2(y_list[1] - y_list[0], x_list[1] - x_list[0])
                worldPos_init.attrib = {
                    'x': '%.16e' % x_list[0], 'y': '%.16e' % y_list[0], 'z': '%.16e' % 0, 'h': '%.16e' % heading, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
                # Stroy部分对车辆动作（轨迹跟随）的设置
                '''
                OpenSCENARIO中通过StoryBoard展现场景的机制：
                story下设置各个车辆的动作集Act，每个Act下定义车辆对应的操作集ManeuverGroup及其触发器StartTrigger
                ManeuverGroup下定义该操作集的执行者Actor及对应的事件Event
                Event下定义具体的车辆动作Action及其触发器StartTrigger
                【对于一个Action，只有动作集Act的触发器触发并且对应Event的触发器也触发，才会执行该动作Action】
                【Act的触发要早于Event，否则仿真将出错，故下面Act的触发时间为0，Event触发时间后移一帧0.05】
                '''
                act = ET.SubElement(story, 'Act')
                act.attrib = {'name': 'Act_Ego'}
                '''车辆操作组ManeuverGroup设置'''
                maneuGroup = ET.SubElement(act, 'ManeuverGroup')
                maneuGroup.attrib = {'maximumExecutionCount': '1', 'name': 'Sequence_Ego'}
                actor = ET.SubElement(maneuGroup, 'Actors')  # 操作执行者设置
                actor.attrib = {'selectTriggeringEntities': 'false'}
                entityRef = ET.SubElement(actor, 'EntityRef')
                entityRef.attrib = {'entityRef': 'Ego'}
                maneuver = ET.SubElement(maneuGroup, 'Maneuver')  # 具体操作设置（通过事件Event的触发）
                maneuver.attrib = {'name': 'Maneuver1'}
                event = ET.SubElement(maneuver, 'Event')
                event.attrib = {'name': 'Event1', 'priority': 'overwrite'}
                action = ET.SubElement(event, 'Action')  # Event下定义的具体车辆动作
                action.attrib = {'name': 'Action1'}
                privateAction_story = ET.SubElement(action, 'PrivateAction')
                privateAction_story.attrib = {}
                routAction = ET.SubElement(privateAction_story, 'RoutingAction')
                routAction.attrib = {}
                followTraAction = ET.SubElement(routAction, 'FollowTrajectoryAction')  # 路径行为模型为轨迹跟随
                followTraAction.attrib = {}
                trajectory = ET.SubElement(followTraAction, 'Trajectory')  # 轨迹跟随的具体轨迹设置
                trajectory.attrib = {'name': 'Trajectory_Ego', 'closed': 'false'}
                shape = ET.SubElement(trajectory, 'Shape')  # 轨迹线型设置（轨迹点之间相连的方式）
                shape.attrib = {}
                polyline = ET.SubElement(shape, 'Polyline')  # 直线相连，highD轨迹为25HZ的采样
                polyline.attrib = {}
                for i in range(len(x_list)):  # 批量填充csv场景中的轨迹点（去除掉初始化点）
                    x = x_list[i]
                    y = y_list[i]
                    vertex = ET.SubElement(polyline, 'Vertex')
                    vertex.attrib = {'time': str(sample * (i))}
                    position_story = ET.SubElement(vertex, 'Position')
                    position_story.attrib = {}
                    worldPos_story = ET.SubElement(position_story, 'WorldPosition')
                    if i == len(x_list) - 1:  # 最后一个轨迹点
                        if x_list[0] < x_list[-1]:  # 下行方向
                            worldPos_story.attrib = {
                                'x': '%.16e' % x, 'y': '%.16e' % y, 'z': '%.16e' % 0, 'h': '%.16e' % 0, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
                        else:  # 上行方向
                            worldPos_story.attrib = {
                                'x': '%.16e' % x, 'y': '%.16e' % y, 'z': '%.16e' % 0, 'h': '%.16e' % math.radians(180), 'p': '%.16e' % 0, 'r': '%.16e' % 0}
                    else:
                        heading = math.atan2(y_list[i + 1] - y_list[i], x_list[i + 1] - x_list[i])
                        worldPos_story.attrib = {
                            'x': '%.16e' % x, 'y': '%.16e' % y, 'z': '%.16e' % 0, 'h': '%.16e' % heading, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
                timeRef = ET.SubElement(followTraAction, 'TimeReference')  # 轨迹跟随的时间设置
                timeRef.attrib = {}
                timing = ET.SubElement(timeRef, 'Timing')  # 选择绝对时间，（不能选择相对事件触发的时间，为保证Act先触发，Event触发点延后了0.03秒）
                timing.attrib = {'domainAbsoluteRelative': 'absolute', 'scale': '1.0', 'offset': '0.0'}
                trajecFolloeMode = ET.SubElement(followTraAction, 'TrajectoryFollowingMode')
                trajecFolloeMode.attrib = {'followingMode': 'follow'}
                startTrig_event = ET.SubElement(event, 'StartTrigger')  # Event的触发器StartTrigger
                startTrig_event.attrib = {}
                conditionGroup_event = ET.SubElement(startTrig_event, 'ConditionGroup')
                conditionGroup_event.attrib = {}
                condition_event = ET.SubElement(conditionGroup_event, 'Condition')
                condition_event.attrib = {'name': '', 'delay': '0', 'conditionEdge': 'rising'}  # 触发机制为rising，即condi由0至1时触发
                byValueCondi_event = ET.SubElement(condition_event, 'ByValueCondition')  # 通过变量值判断条件
                byValueCondi_event.attrib = {}
                simulationTimeCondi_event = ET.SubElement(byValueCondi_event, 'SimulationTimeCondition')  # 基于仿真时间触发
                simulationTimeCondi_event.attrib = {'value': '0.03', 'rule': 'greaterThan'}
                '''车辆动作集Act的触发器设置'''
                startTrig_act = ET.SubElement(act, 'StartTrigger')  # 动作集Act触发器设置
                startTrig_act.attrib = {}
                conditionGroup_act = ET.SubElement(startTrig_act, 'ConditionGroup')
                conditionGroup_act.attrib = {}
                condition_act = ET.SubElement(conditionGroup_act, 'Condition')
                condition_act.attrib = {'name': '', 'delay': '0', 'conditionEdge': 'rising'}
                byValueCondi_act = ET.SubElement(condition_act, 'ByValueCondition')
                byValueCondi_act.attrib = {}
                simulationTimeCondi_act = ET.SubElement(byValueCondi_act, 'SimulationTimeCondition')
                simulationTimeCondi_act.attrib = {'value': '0', 'rule': 'greaterThan'}
            else:
                v_0 = (x_list[2] - x_list[0]) / float(2 * sample)
                heading = math.atan2(y_list[1] - y_list[0], x_list[1] - x_list[0])
                ego_info1 = ET.Comment('Information of the ego vehicle will be hidden, and its initial state and driving task will be explained in the comments below')
                ego_info2 = ET.Comment('[Initial State] v_init = ' + str(abs(v_0)) + ', x_init = ' + str(x_list[0]) + ', y_init = ' + str(y_list[0]) + ', heading_init = ' + str(heading))
                ego_info3 = ET.Comment('[Driving Task] x_target = (' + str(x_list[-1] - 5) + ', ' + str(x_list[-1] + 5) + '), y_target = (' + str(y_list[-1] - 1) + ', ' + str(y_list[-1] + 1) + ')')
                private_ego.append(ego_info1)
                private_ego.append(ego_info2)
                private_ego.append(ego_info3)
        else:  # 非ego
            # 申明Entities及对应属性
            scenObj = ET.SubElement(entity, 'ScenarioObject')
            scenObj.attrib = {'name': str('A' + str(count))}
            veh = ET.SubElement(scenObj, 'Vehicle')
            if flag == 0:
                veh.attrib = {'name': 'Default_car', 'vehicleCategory': 'car'}
            elif flag == 1 and count == 1:
                veh.attrib = {'name': 'Default_car', 'vehicleCategory': 'car', 'model3d': 'car_white.osgb'}
            boundingbox = ET.SubElement(veh, 'BoundingBox')  # 车辆边框属性设置
            center = ET.SubElement(boundingbox, 'Center')  # 车辆中心在【车辆坐标系】中的坐标
            center.attrib = {'x': '%.16e' % 1.5, 'y': '%.16e' % 0, 'z': '%.16e' % 0.9}
            dimension = ET.SubElement(boundingbox, 'Dimensions')
            dimension.attrib = {'width': '%.16e' % 2.1, 'length': '%.16e' % 4.5, 'height': '%.16e' % 1.8}
            performance = ET.SubElement(veh, 'Performance')
            performance.attrib = {'maxSpeed': "200", 'maxAcceleration': "200", 'maxDeceleration': "10.0"}
            axles = ET.SubElement(veh, 'Axles')
            axles.attrib = {}
            front = ET.SubElement(axles, 'FrontAxle')
            front.attrib = {'maxSteering': "0.5", 'wheelDiameter': "0.5", 'trackWidth': "1.75", 'positionX': "2.8", 'positionZ': "0.25"}
            rear = ET.SubElement(axles, 'RearAxle')
            rear.attrib = {'maxSteering': "0.0", 'wheelDiameter': "0.5", 'trackWidth': "1.75", 'positionX': "0.0", 'positionZ': "0.25"}
            controller = ET.SubElement(scenObj, 'ObjectController')
            controller.attrib = {}
            # Init部分对车辆的初始化
            private = ET.Element('Private', {'entityRef': str('A' + str(count))})  # 申明专属对象
            actions.append(private)

            privateAction1_init = ET.SubElement(private, 'PrivateAction')  # 初始化专属动作1（速度）
            privateAction1_init.attrib = {}
            longitAction = ET.SubElement(privateAction1_init, 'LongitudinalAction')
            longitAction.attrib = {}
            speedAction = ET.SubElement(longitAction, 'SpeedAction')
            speedAction.attrib = {}
            speedAcDy = ET.SubElement(speedAction, 'SpeedActionDynamics')
            speedAcDy.attrib = {'dynamicsShape': 'step', 'value': '0', 'dynamicsDimension': 'time'}
            speedAcTar = ET.SubElement(speedAction, 'SpeedActionTarget')
            speedAcTar.attrib = {}
            absTarSpeed = ET.SubElement(speedAcTar, 'AbsoluteTargetSpeed')
            v_0 = (x_list[2] - x_list[0]) / float(2 * sample)
            absTarSpeed.attrib = {'value': '%.16e' % v_0}

            privateAction2_init = ET.SubElement(private, 'PrivateAction')  # 初始化专属动作2（位置）
            privateAction2_init.attrib = {}
            telepAction = ET.SubElement(privateAction2_init, 'TeleportAction')
            telepAction.attrib = {}
            position_init = ET.SubElement(telepAction, 'Position')
            position_init.attrib = {}
            worldPos_init = ET.SubElement(position_init, 'WorldPosition')  # 采用全局世界坐标对车辆进行定位
            heading = math.atan2(y_list[1] - y_list[0], x_list[1] - x_list[0])
            worldPos_init.attrib = {
                'x': '%.16e' % x_list[0], 'y': '%.16e' % y_list[0], 'z': '%.16e' % 0, 'h': '%.16e' % heading, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
            # Stroy部分对车辆动作（轨迹跟随）的设置
            act = ET.SubElement(story, 'Act')
            act.attrib = {'name': str('Act_' + 'A' + str(count))}
            '''车辆操作组ManeuverGroup设置'''
            maneuGroup = ET.SubElement(act, 'ManeuverGroup')
            maneuGroup.attrib = {'maximumExecutionCount': '1', 'name': str('Squence_' + 'A' + str(count))}
            actor = ET.SubElement(maneuGroup, 'Actors')  # 操作执行者设置
            actor.attrib = {'selectTriggeringEntities': 'false'}
            entityRef = ET.SubElement(actor, 'EntityRef')
            entityRef.attrib = {'entityRef': str('A' + str(count))}
            maneuver = ET.SubElement(maneuGroup, 'Maneuver')  # 具体操作设置（通过事件Event的触发）
            maneuver.attrib = {'name': 'Maneuver1'}
            event = ET.SubElement(maneuver, 'Event')
            event.attrib = {'name': 'Event1', 'priority': 'overwrite'}
            action = ET.SubElement(event, 'Action')  # Event下定义的具体车辆动作
            action.attrib = {'name': 'Action1'}
            privateAction_story = ET.SubElement(action, 'PrivateAction')
            privateAction_story.attrib = {}
            routAction = ET.SubElement(privateAction_story, 'RoutingAction')
            routAction.attrib = {}
            followTraAction = ET.SubElement(routAction, 'FollowTrajectoryAction')  # 路径行为模型为轨迹跟随
            followTraAction.attrib = {}
            trajectory = ET.SubElement(followTraAction, 'Trajectory')  # 轨迹跟随的具体轨迹设置
            trajectory.attrib = {'name': str('Trajectory_' + 'A' + str(count)), 'closed': 'false'}
            shape = ET.SubElement(trajectory, 'Shape')  # 轨迹线型设置（轨迹点之间相连的方式）
            shape.attrib = {}
            polyline = ET.SubElement(shape, 'Polyline')  # 直线相连，highD轨迹为25HZ的采样
            polyline.attrib = {}
            for i in range(len(x_list)):  # 批量填充csv场景中的轨迹点（去除掉初始化点）
                x = x_list[i]
                y = y_list[i]
                vertex = ET.SubElement(polyline, 'Vertex')
                vertex.attrib = {'time': str(sample * (i))}
                position_story = ET.SubElement(vertex, 'Position')
                position_story.attrib = {}
                worldPos_story = ET.SubElement(position_story, 'WorldPosition')
                if i == len(x_list) - 1:  # 最后一个轨迹点
                    if x_list[0] < x_list[-1]:  # 下行方向
                        worldPos_story.attrib = {
                            'x': '%.16e' % x, 'y': '%.16e' % y, 'z': '%.16e' % 0, 'h': '%.16e' % 0, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
                    else:  # 上行方向
                        worldPos_story.attrib = {
                            'x': '%.16e' % x, 'y': '%.16e' % y, 'z': '%.16e' % 0, 'h': '%.16e' % math.radians(180), 'p': '%.16e' % 0, 'r': '%.16e' % 0}
                else:
                    heading = math.atan2(y_list[i + 1] - y_list[i], x_list[i + 1] - x_list[i])
                    worldPos_story.attrib = {
                        'x': '%.16e' % x, 'y': '%.16e' % y, 'z': '%.16e' % 0, 'h': '%.16e' % heading, 'p': '%.16e' % 0, 'r': '%.16e' % 0}
            timeRef = ET.SubElement(followTraAction, 'TimeReference')  # 轨迹跟随的时间设置
            timeRef.attrib = {}
            timing = ET.SubElement(timeRef, 'Timing')
            timing.attrib = {'domainAbsoluteRelative': 'absolute', 'scale': '1.0', 'offset': '0.0'}
            trajecFolloeMode = ET.SubElement(followTraAction, 'TrajectoryFollowingMode')
            trajecFolloeMode.attrib = {'followingMode': 'follow'}
            startTrig_event = ET.SubElement(event, 'StartTrigger')  # Event的触发器StartTrigger
            startTrig_event.attrib = {}
            conditionGroup_event = ET.SubElement(startTrig_event, 'ConditionGroup')
            conditionGroup_event.attrib = {}
            condition_event = ET.SubElement(conditionGroup_event, 'Condition')
            condition_event.attrib = {'name': '', 'delay': '0', 'conditionEdge': 'rising'}  # 触发机制为rising，即condi由0至1时触发
            byValueCondi_event = ET.SubElement(condition_event, 'ByValueCondition')  # 通过变量值判断条件
            byValueCondi_event.attrib = {}
            simulationTimeCondi_event = ET.SubElement(byValueCondi_event, 'SimulationTimeCondition')  # 基于仿真时间触发
            simulationTimeCondi_event.attrib = {'value': '0.03', 'rule': 'greaterThan'}
            '''车辆动作集Act的触发器设置'''
            startTrig_act = ET.SubElement(act, 'StartTrigger')  # 动作集Act触发器设置
            startTrig_act.attrib = {}
            conditionGroup_act = ET.SubElement(startTrig_act, 'ConditionGroup')
            conditionGroup_act.attrib = {}
            condition_act = ET.SubElement(conditionGroup_act, 'Condition')
            condition_act.attrib = {'name': '', 'delay': '0', 'conditionEdge': 'rising'}
            byValueCondi_act = ET.SubElement(condition_act, 'ByValueCondition')
            byValueCondi_act.attrib = {}
            simulationTimeCondi_act = ET.SubElement(byValueCondi_act, 'SimulationTimeCondition')
            simulationTimeCondi_act.attrib = {'value': '0', 'rule': 'greaterThan'}
        count += 1

    # 设置整个StoryBoard的停止器StopTrigger
    conditionGroup_stop = ET.SubElement(stoptrigger, 'ConditionGroup')
    conditionGroup_stop.attrib = {}
    condition_stop = ET.SubElement(conditionGroup_stop, 'Condition')
    condition_stop.attrib = {'name': '', 'delay': '0', 'conditionEdge': 'rising'}  # 触发机制为rising，即condi由0至1时触发
    byValueCondi_stop = ET.SubElement(condition_stop, 'ByValueCondition')  # 通过变量值判断条件
    byValueCondi_stop.attrib = {}
    simulationTimeCondi_stop = ET.SubElement(byValueCondi_stop, 'SimulationTimeCondition')  # 基于仿真时间触发
    simulationTimeCondi_stop.attrib = {'value': str(whole_time + 0.025), 'rule': 'greaterThan'}

    tree = ET.ElementTree(root)
    pretty_xml(root, '\t', '\n')  # 执行美化方法
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    return


if __name__ == "__main__":
    # 反归一化泛化场景
    x1_min, x1_max = 0.0, 235.5454259747434
    y1_min, y1_max = -3.679218247347115, 2.5978529242443447
    x2_min, x2_max = -19.260000000000048, 385.4497701569932
    y2_min, y2_max = -5.030000000000001, 6.909999999999997
    x3_min, x3_max = 8.329999999999998, 413.4503592184133
    y3_min, y3_max = -3.460082480103437, 3.69
    lims = [x1_min, x1_max, y1_min, y1_max, x2_min, x2_max, y2_min, y2_max, x3_min, x3_max, y3_min, y3_max]
    sampleScene = Repair_cutin(fusion_path='./processed_by_zwt/samples/data_modify_twice_2.5_4.npy', lims=lims)
    # sampleScene = Repair_cutin(fusion_path='./processed_by_zwt/input_data/cutin3_xy_singleD_proceseed_norm_train.npy', lims=lims)
    min_x, max_x, min_y, max_y = [], [], [], []
    for i in range(sampleScene.fusion_re.shape[0]):
        data = sampleScene.fusion_re[i, :, :]
        min_x.append(min([min(data[:, 2 * j]) for j in range(3)]))
        max_x.append(max([max(data[:, 2 * j]) for j in range(3)]))
        min_y.append(min([min(data[:, 2 * j + 1]) for j in range(3)]))
        max_y.append(max([max(data[:, 2 * j + 1]) for j in range(3)]))
    # print(min(min_x), max(max_x), min(min_y), max(max_y))

    # 转换为xosc
    base_xodr = r'./processed_by_zwt/input_data/base.xodr'
    for id_ in tqdm(range(sampleScene.fusion_re.shape[0])):
        # 新建场景文件夹
        dir_name = 'vae_wgan_sample_' + str(id_)
        xodr_name = 'vae_wgan_sample_' + str(id_) + '.xodr'
        cur = r'../openX_paper/sample_paper_4'
        target_path = cur + "/" + dir_name
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            # 复制场景地图并更名
            xodr_path_old = target_path + '/' + 'base.xodr'
            xodr_path_new = target_path + '/' + 'vae_wgan_sample_' + str(id_) + '.xodr'
            shutil.copyfile(base_xodr, xodr_path_old)
            os.rename(xodr_path_old, xodr_path_new)
        output_path_st = target_path + '/vae_wgan_sample_' + str(id_) + '.xosc'
        output_path_exam = target_path + '/vae_wgan_sample_' + str(id_) + '_exam' + '.xosc'
        data = sampleScene.fusion_re[id_, :, :]
        # xosc_write(data, xodr_name, output_path_st)
        xosc_write(data, xodr_name, output_path_exam, is_exam=True)