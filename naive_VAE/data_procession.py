import os
import pywt
import numpy as np
import pandas as pd
from tqdm import tqdm
from gekko import GEKKO
import matplotlib.pyplot as plt
import tracemalloc


def plot_outlier_adjacent_trj(series_in, outlier_pos_in, first_pos_in, last_pos_in, segment_id_in, veh_id_in, start_time_in, comparison_label, flag):
    # plot the adjacent trajectory of the outlier (20 points)
    correctness_dir = ['_x', '_y']
    outlier_time = round(start_time_in + outlier_pos_in * 0.04, 2)
    included_index = np.arange(first_pos_in, last_pos_in + 1, dtype=int)
    outlier_trj = series_in.loc[included_index, :]
    outlier_trj.loc[:, 'local_time'] = np.array(included_index) * 0.04 + start_time_in
    plt.subplot(3, 1, 1)
    plt.plot(outlier_trj['local_time'], outlier_trj[('cumu_dis' + correctness_dir[flag])], '-*k', linewidth=0.25, label='Original', markersize=1.5)
    if comparison_label == 1:
        plt.plot(outlier_trj['local_time'], outlier_trj[('remove_outlier_cumu_dis' + correctness_dir[flag])], '-m', linewidth=0.25, label='Outliers Removed')
        plt.legend(prop={'size': 6})
        trj_title = 'Scenario ' + str(int(segment_id_in)) + ' Vehicle' + str(
            int(veh_id_in)) + ' Direction' + correctness_dir[flag] + ' Outlier at Time ' + str(outlier_time) + ' Removing'
    else:
        trj_title = 'Scenario ' + str(int(segment_id_in)) + ' Vehicle' + str(
            int(veh_id_in)) + ' Direction' + correctness_dir[flag] + ' Outlier at Time ' + str(outlier_time) + ' Pattern'
    plt.ylabel('Position (m)')
    plt.title(trj_title)
    plt.subplot(3, 1, 2)
    plt.plot(outlier_trj['local_time'], outlier_trj[('speed' + correctness_dir[flag])], '-*k', linewidth=0.5, label='Original', markersize=1.5)
    if comparison_label == 1:
        plt.plot(outlier_trj['local_time'], outlier_trj[('remove_outlier_speed' + correctness_dir[flag])], '-m', linewidth=0.5, label='Outliers Removed')
        plt.legend(prop={'size': 6})
    plt.ylabel('Speed (m/s)')
    if not flag:
        plt.ylim([0, 50])
    else:
        plt.ylim([-10, 10])
    plt.subplot(3, 1, 3)
    plt.plot(outlier_trj['local_time'], outlier_trj[('accer' + correctness_dir[flag])], '-*k', linewidth=0.5, label='Original', markersize=1.5)
    if comparison_label == 1:
        plt.plot(outlier_trj['local_time'], outlier_trj[('remove_outlier_accer' + correctness_dir[flag])], '-m', linewidth=0.5, label='Outliers Removed')
        plt.legend(prop={'size': 6})
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s2)')
    if not flag:
        plt.ylim([-15, 15])
    else:
        plt.ylim([-3, 3])
    if not os.path.exists('figure_save/trajectory_process/outlier_pattern_and_removing'):
        os.makedirs('figure_save/trajectory_process/outlier_pattern_and_removing')
    trj_save_title = 'figure_save/trajectory_process/outlier_pattern_and_removing/' + trj_title + '.png'
    plt.savefig(trj_save_title, dpi=600)
    plt.close('all')  # test


def before_and_after_remove_outlier_plot(trj_in):
    current_seg_id = trj_in['segment_id'].iloc[0]
    follower_id_in = trj_in['local_veh_id'].iloc[0]
    correctness_dir = ['_x', '_y']
    for i in range(2):
        plt.subplot(3, 1, 1)
        plt.plot(trj_in['local_time'], trj_in['position' + correctness_dir[i]], '--k', linewidth=0.25, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_pos' + correctness_dir[i]], '-m', linewidth=0.25, label='Outliers Removed')
        plt.ylabel('Position (m)')
        plt.legend(prop={'size': 6})
        trj_title = 'Scenario ' + str(int(current_seg_id)) + ' Vehicle' + str(
            int(follower_id_in)) + ' Direction' + correctness_dir[i] + ' Before and After Removing Outliers'
        plt.title(trj_title)
        plt.subplot(3, 1, 2)
        plt.plot(trj_in['local_time'], trj_in['speed' + correctness_dir[i]], '--k', linewidth=0.5, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_speed' + correctness_dir[i]], '-m', linewidth=0.5, label='Outliers Removed')
        plt.ylabel('Speed (m/s)')
        plt.legend(prop={'size': 6})
        if not i:
            plt.ylim([0, 50])
        else:
            plt.ylim([-10, 10])
        plt.subplot(3, 1, 3)
        plt.plot(trj_in['local_time'], trj_in['accer' + correctness_dir[i]], '--k', linewidth=0.5, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_accer' + correctness_dir[i]], '-m', linewidth=0.5, label='Outliers Removed')
        plt.legend(prop={'size': 6})
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s2)')
        if not i:
            plt.ylim([-15, 15])
        else:
            plt.ylim([-5, 5])
        if not os.path.exists('figure_save/trajectory_process/before_and_after_remove_outlier_plot'):
            os.makedirs('figure_save/trajectory_process/before_and_after_remove_outlier_plot')
        trj_save_title = 'figure_save/trajectory_process/before_and_after_remove_outlier_plot/' + trj_title + '.png'
        plt.savefig(trj_save_title, dpi=600)
        plt.close('all')


def before_and_after_filtering_plot(trj_in):
    current_seg_id = trj_in['segment_id'].iloc[0]
    follower_id_in = trj_in['local_veh_id'].iloc[0]
    correctness_dir = ['_x', '_y']
    for i in range(2):
        plt.subplot(3, 1, 1)
        plt.plot(trj_in['local_time'], trj_in['position' + correctness_dir[i]], '--k', linewidth=0.25, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_pos' + correctness_dir[i]], '-m', linewidth=0.25, label='Outliers Removed')
        plt.plot(trj_in['local_time'], trj_in['filter_pos' + correctness_dir[i]], '-*g', linewidth=0.25, label='Outliers Removed + Filtering', markersize=0.5)
        plt.ylabel('Position (m)')
        plt.legend(prop={'size': 6})
        trj_title = 'Scenario ' + str(int(current_seg_id)) + ' Vehicle' + str(
            int(follower_id_in)) + ' Direction' + correctness_dir[i] + ' Before and After Filtering'
        plt.title(trj_title)
        plt.subplot(3, 1, 2)
        plt.plot(trj_in['local_time'], trj_in['speed' + correctness_dir[i]], '--k', linewidth=0.25, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_speed' + correctness_dir[i]], '-m', linewidth=0.25, label='Outliers Removed')
        plt.plot(trj_in['local_time'], trj_in['filter_speed' + correctness_dir[i]], '-*g', linewidth=0.25, label='Outliers Removed + Filtering', markersize=0.5)
        plt.ylabel('Speed (m/s)')
        plt.legend(prop={'size': 6})
        if not i:
            plt.ylim([0, 50])
        else:
            plt.ylim([-10, 10])
        plt.subplot(3, 1, 3)
        plt.plot(trj_in['local_time'], trj_in['accer' + correctness_dir[i]], '--k', linewidth=0.25, label='Original')
        plt.plot(trj_in['local_time'], trj_in['remove_outlier_accer' + correctness_dir[i]], '-m', linewidth=0.25, label='Outliers Removed')
        plt.plot(trj_in['local_time'], trj_in['filter_accer' + correctness_dir[i]], '-*g', linewidth=0.25, label='Outliers Removed + Filtering', markersize=0.5)
        plt.legend(prop={'size': 6})
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s2)')
        if not i:
            plt.ylim([-15, 15])
        else:
            plt.ylim([-5, 5])
        if not os.path.exists('figure_save/trajectory_process/before_and_after_filtering_plot'):
            os.makedirs('figure_save/trajectory_process/before_and_after_filtering_plot')
        trj_save_title = 'figure_save/trajectory_process/before_and_after_filtering_plot/' + trj_title + '.png'
        plt.savefig(trj_save_title, dpi=600)
        plt.close('all')


def cf_paired_trj_plot(leader_trj_in, follower_trj_in, av_label):
    # av_label is to determine whether av is leader or follower (0 for follower, 1 for leader, 2 for non-av)
    # the format of the trajectory is pandas dataframe
    # for av_label: 0 means AV-HV, 1 means HV-AV, 2 means HV-HV
    current_segment_id = int(leader_trj_in['segment_id'].iloc[0])
    current_leader_id = int(leader_trj_in['local_veh_id'].iloc[0])
    current_follower_id = int(follower_trj_in['local_veh_id'].iloc[0])
    if av_label == 0:
        follower_line = '-r'
        leader_line = '--b'
        follower_label = 'Cutin Challenger ' + str(current_follower_id)
        leader_label = 'Ego ' + str(current_leader_id)
        # trj_title = 'Cutin Scenario ' + str(current_segment_id)
        # trj_save_title = 'figure_save/trajectory_process/position_time_plot/' + 'Segment_' + str(
        #     current_segment_id) + '_' + trj_title + '_position_time_plot.png'
    else:
        follower_line = '-b'
        leader_line = '--b'
        follower_label = 'HV Follower'
        leader_label = 'HV Leader'
        trj_title = 'HV' + str(current_follower_id) + '-HV' + str(current_leader_id)
        if not os.path.exists('figure_save/trajectory_process/position_time_plot'):
            os.makedirs('figure_save/trajectory_process/position_time_plot')
        trj_save_title = 'figure_save/trajectory_process/position_time_plot/' + 'Segment_' + str(
            current_segment_id) + '_' + trj_title + '_position_time_plot.png'
    correctness_dir = ['_x', '_y']
    for i in range(2):
        plt.subplot(3, 1, 1)
        plt.plot(follower_trj_in['local_time'], follower_trj_in['filter_pos' + correctness_dir[i]], follower_line, linewidth=0.5, label=follower_label)
        plt.plot(leader_trj_in['local_time'], leader_trj_in['filter_pos' + correctness_dir[i]], leader_line, linewidth=0.5, label=leader_label)
        plt.ylabel('Position (m)')
        plt.legend(prop={'size': 6})
        trj_title = 'Cutin Scenario ' + str(current_segment_id) + ' Direction' + correctness_dir[i]
        trj_save_title = 'figure_save/trajectory_process/position_time_plot/' + trj_title + '_position_time_plot.png'
        plt.title(trj_title)
        plt.subplot(3, 1, 2)
        plt.plot(follower_trj_in['local_time'], follower_trj_in['filter_speed' + correctness_dir[i]], follower_line, linewidth=0.5, label=follower_label)
        plt.plot(leader_trj_in['local_time'], leader_trj_in['filter_speed' + correctness_dir[i]], leader_line, linewidth=0.5, label=leader_label)
        plt.ylabel('Speed (m/s)')
        plt.legend(prop={'size': 6})
        if not i:
            plt.ylim([0, 50])
        else:
            plt.ylim([-10, 10])
        plt.subplot(3, 1, 3)
        plt.plot(follower_trj_in['local_time'], follower_trj_in['filter_accer' + correctness_dir[i]], follower_line, linewidth=0.5, label=follower_label)
        plt.plot(leader_trj_in['local_time'], leader_trj_in['filter_accer' + correctness_dir[i]], leader_line, linewidth=0.5, label=leader_label)
        plt.legend(prop={'size': 6})
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s2)')
        if not i:
            plt.ylim([-8, 5])
        else:
            plt.ylim([-5, 5])
        plt.savefig(trj_save_title, dpi=600)
        plt.close('all')


def update_speed_and_accer(series_in, filter_label):
    # this function calculate the speed, accelearation, jerk based on position
    # series_in is the same format as  coord_series_in
    # output is series_in with updated speed and accer

    if filter_label == 1:
        current_cumu_dis_x = 'filter_cumu_dis_x'
        current_speed_x = 'filter_speed_x'
        current_accer_x = 'filter_accer_x'
        current_cumu_dis_y = 'filter_cumu_dis_y'
        current_speed_y = 'filter_speed_y'
        current_accer_y = 'filter_accer_y'
    elif filter_label == 0:
        current_cumu_dis_x = 'cumu_dis_x'
        current_speed_x = 'speed_x'
        current_accer_x = 'accer_x'
        current_jerk_x = 'jerk_x'
        current_cumu_dis_y = 'cumu_dis_y'
        current_speed_y = 'speed_y'
        current_accer_y = 'accer_y'
        current_jerk_y = 'jerk_y'
    elif filter_label == 2:
        current_cumu_dis_x = 'remove_outlier_cumu_dis_x'
        current_speed_x = 'remove_outlier_speed_x'
        current_accer_x = 'remove_outlier_accer_x'
        current_jerk_x = 'remove_outlier_jerk_x'
        current_cumu_dis_y = 'remove_outlier_cumu_dis_y'
        current_speed_y = 'remove_outlier_speed_y'
        current_accer_y = 'remove_outlier_accer_y'
        current_jerk_y = 'remove_outlier_jerk_y'

    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_speed_x] = float(
                series_in.at[i + 2, current_cumu_dis_x] - series_in.at[i, current_cumu_dis_x]) / (float(0.08))
            series_in.at[i, current_speed_y] = float(
                series_in.at[i + 2, current_cumu_dis_y] - series_in.at[i, current_cumu_dis_y]) / (float(0.08))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_speed_x] = float(
                series_in.at[i, current_cumu_dis_x] - series_in.at[i - 2, current_cumu_dis_x]) / (float(0.08))
            series_in.at[i, current_speed_y] = float(
                series_in.at[i, current_cumu_dis_y] - series_in.at[i - 2, current_cumu_dis_y]) / (float(0.08))
        else:
            series_in.at[i, current_speed_x] = float(
                series_in.at[i + 1, current_cumu_dis_x] - series_in.at[i - 1, current_cumu_dis_x]) / (float(0.08))
            series_in.at[i, current_speed_y] = float(
                series_in.at[i + 1, current_cumu_dis_y] - series_in.at[i - 1, current_cumu_dis_y]) / (float(0.08))

    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_accer_x] = float(
                series_in.at[i + 2, current_speed_x] - series_in.at[i, current_speed_x]) / (float(0.08))
            series_in.at[i, current_accer_y] = float(
                series_in.at[i + 2, current_speed_y] - series_in.at[i, current_speed_y]) / (float(0.08))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_accer_x] = float(
                series_in.at[i, current_speed_x] - series_in.at[i - 2, current_speed_x]) / (float(0.08))
            series_in.at[i, current_accer_y] = float(
                series_in.at[i, current_speed_y] - series_in.at[i - 2, current_speed_y]) / (float(0.08))
        else:
            series_in.at[i, current_accer_x] = float(
                series_in.at[i + 1, current_speed_x] - series_in.at[i - 1, current_speed_x]) / (float(0.08))
            series_in.at[i, current_accer_y] = float(
                series_in.at[i + 1, current_speed_y] - series_in.at[i - 1, current_speed_y]) / (float(0.08))

    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_jerk_x] = float(
                series_in.at[i + 2, current_accer_x] - series_in.at[i, current_accer_x]) / (float(0.08))
            series_in.at[i, current_jerk_y] = float(
                series_in.at[i + 2, current_accer_y] - series_in.at[i, current_accer_y]) / (float(0.08))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_jerk_x] = float(
                series_in.at[i, current_accer_x] - series_in.at[i - 2, current_accer_x]) / (float(0.08))
            series_in.at[i, current_jerk_y] = float(
                series_in.at[i, current_accer_y] - series_in.at[i - 2, current_accer_y]) / (float(0.08))
        else:
            series_in.at[i, current_jerk_x] = float(
                series_in.at[i + 1, current_accer_x] - series_in.at[i - 1, current_accer_x]) / (float(0.08))
            series_in.at[i, current_jerk_y] = float(
                series_in.at[i + 1, current_accer_y] - series_in.at[i - 1, current_accer_y]) / (float(0.08))
    return series_in


def speed_based_update_distance_and_accer(series_in):
    # this function calculate the distance, acceleration and jerk based on speed (for speed-based data)
    # series_in is the same format as  coord_series_in
    # output is series_in with updated speed and accer

    current_cumu_dis_x = 'speed_based_cumu_dis_x'
    current_speed_x = 'speed_based_speed_x'
    current_accer_x = 'speed_based_accer_x'
    current_jerk_x = 'speed_based_jerk_x'
    current_cumu_dis_y = 'speed_based_cumu_dis_y'
    current_speed_y = 'speed_based_speed_y'
    current_accer_y = 'speed_based_accer_y'
    current_jerk_y = 'speed_based_jerk_y'

    for i in range(1, len(series_in['global_center_x'])):
        if i == 1:
            series_in.loc[0, current_cumu_dis_x] = 0
            series_in.loc[i, current_cumu_dis_x] = series_in.loc[i - 1, current_cumu_dis_x] + (
                series_in.loc[i, current_speed_x] + series_in.loc[i - 1, current_speed_x]) * 0.5 * 0.04
            series_in.loc[0, current_cumu_dis_y] = 0
            series_in.loc[i, current_cumu_dis_y] = series_in.loc[i - 1, current_cumu_dis_y] + (
                series_in.loc[i, current_speed_y] + series_in.loc[i - 1, current_speed_y]) * 0.5 * 0.04
        else:
            series_in.loc[i, current_cumu_dis_x] = series_in.loc[i - 1, current_cumu_dis_x] + (
                series_in.loc[i, current_speed_x] + series_in.loc[i - 1, current_speed_x]) * 0.5 * 0.04
            series_in.loc[i, current_cumu_dis_y] = series_in.loc[i - 1, current_cumu_dis_y] + (
                series_in.loc[i, current_speed_y] + series_in.loc[i - 1, current_speed_y]) * 0.5 * 0.04

    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_accer_x] = float(
                series_in.at[i + 2, current_speed_x] - series_in.at[i, current_speed_x]) / (float(0.08))
            series_in.at[i, current_accer_y] = float(
                series_in.at[i + 2, current_speed_y] - series_in.at[i, current_speed_y]) / (float(0.08))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_accer_x] = float(
                series_in.at[i, current_speed_x] - series_in.at[i - 2, current_speed_x]) / (float(0.08))
            series_in.at[i, current_accer_y] = float(
                series_in.at[i, current_speed_y] - series_in.at[i - 2, current_speed_y]) / (float(0.08))
        else:
            series_in.at[i, current_accer_x] = float(
                series_in.at[i + 1, current_speed_x] - series_in.at[i - 1, current_speed_x]) / (float(0.08))
            series_in.at[i, current_accer_y] = float(
                series_in.at[i + 1, current_speed_y] - series_in.at[i - 1, current_speed_y]) / (float(0.08))

    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_jerk_x] = float(
                series_in.at[i + 2, current_accer_x] - series_in.at[i, current_accer_x]) / (float(0.08))
            series_in.at[i, current_jerk_y] = float(
                series_in.at[i + 2, current_accer_y] - series_in.at[i, current_accer_y]) / (float(0.08))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_jerk_x] = float(
                series_in.at[i, current_accer_x] - series_in.at[i - 2, current_accer_x]) / (float(0.08))
            series_in.at[i, current_jerk_y] = float(
                series_in.at[i, current_accer_y] - series_in.at[i - 2, current_accer_y]) / (float(0.08))
        else:
            series_in.at[i, current_jerk_x] = float(
                series_in.at[i + 1, current_accer_x] - series_in.at[i - 1, current_accer_x]) / (float(0.08))
            series_in.at[i, current_jerk_y] = float(
                series_in.at[i + 1, current_accer_y] - series_in.at[i - 1, current_accer_y]) / (float(0.08))
    return series_in


def outlier_removing_optimization_model(initial_state_in, last_state_in, num_points_in):
    max_acc = 5
    min_acc = -8
    total_steps = num_points_in
    first_pos_in = initial_state_in[0]
    first_speed_in = initial_state_in[1]
    first_acc_in = initial_state_in[2]
    last_pos_in = last_state_in[0]
    last_speed_in = last_state_in[1]
    last_acc_in = last_state_in[2]

    time_interval = 0.04

    model = GEKKO(remote=False)
    model.options.SOLVER = 3
    model.options.SCALING = 2
    model.options.MAX_MEMORY = 5
    # model.options.IMODE = 2  # Steady state optimization
    acc = [None] * total_steps  # simulated acceleration
    velocity = [None] * total_steps  # simulated velocity
    pos = [None] * total_steps  # simulated position
    for i in range(total_steps):
        pos[i] = model.Var()
        velocity[i] = model.Var()
        velocity[i].lower = 0
        acc[i] = model.Var(lb=min_acc, ub=max_acc)
    min_sim_acc = model.Var()
    max_sim_acc = model.Var()
    model.Equation(pos[0] == first_pos_in)
    model.Equation(velocity[0] == first_speed_in)
    model.Equation(acc[0] == first_acc_in)
    model.Equation(pos[total_steps - 1] == last_pos_in)
    model.Equation(velocity[total_steps - 1] == last_speed_in)
    model.Equation(acc[total_steps - 1] == last_acc_in)
    for i in range(total_steps):
        if 1 <= i <= total_steps - 1:
            model.Equation(velocity[i] == velocity[i - 1] + acc[i - 1] * time_interval)
            model.Equation(pos[i] == pos[i - 1] + 0.5 * (velocity[i] + velocity[i - 1]) * time_interval)
    for i in range(total_steps):
        model.Equation(min_sim_acc <= acc[i])
        model.Equation(max_sim_acc >= acc[i])
    model.Obj(max_sim_acc - min_sim_acc)
    try:
        model.solve(disp=False)
    except Exception:
        return False
    # solve_time = model.options.SOLVETIME
    # extract values from Gekko type variables
    acc_value = np.zeros(total_steps)
    velocity_value = np.zeros(total_steps)
    pos_value = np.zeros(total_steps)
    for i in range(total_steps):
        acc_value[i] = acc[i].value[0]
        velocity_value[i] = velocity[i].value[0]
        pos_value[i] = pos[i].value[0]
    return pos_value, velocity_value, acc_value


def optimization_based_outlier_removing(series_in, first_pos_in, last_pos_in, min_acc_in, max_acc_in, flag):
    if flag == 0:
        series_status = ['remove_outlier_cumu_dis_x', 'remove_outlier_speed_x', 'remove_outlier_accer_x']
    else:
        series_status = ['remove_outlier_cumu_dis_y', 'remove_outlier_speed_y', 'remove_outlier_accer_y']
    first_point_pos = first_pos_in
    last_point_pos = last_pos_in
    first_point_cumu_dis = series_in.at[first_point_pos, series_status[0]]
    first_point_speed = series_in.at[first_point_pos, series_status[1]]

    if series_in.at[first_point_pos, series_status[2]] <= min_acc_in:
        first_point_acc = min_acc_in
    elif series_in.at[first_point_pos, series_status[2]] >= max_acc_in:
        first_point_acc = max_acc_in
    else:
        first_point_acc = series_in.at[first_point_pos, series_status[2]]
    first_point_state = [first_point_cumu_dis, first_point_speed, first_point_acc]
    last_point_cumu_dis = series_in.at[last_point_pos, series_status[0]]
    last_point_speed = series_in.at[last_point_pos, series_status[1]]
    if series_in.at[last_point_pos, series_status[2]] <= min_acc_in:
        last_point_acc = min_acc_in
    elif series_in.at[last_point_pos, series_status[2]] >= max_acc_in:
        last_point_acc = max_acc_in
    else:
        last_point_acc = series_in.at[last_point_pos, series_status[2]]
    last_point_state = [last_point_cumu_dis, last_point_speed, last_point_acc]

    actual_total_related_points = last_point_pos - first_point_pos + 1
    if outlier_removing_optimization_model(first_point_state, last_point_state, actual_total_related_points):
        pos_result, speed_result, acc_result = outlier_removing_optimization_model(first_point_state, last_point_state, actual_total_related_points)
    else:
        return pd.DataFrame()
    series_in.loc[first_point_pos:last_point_pos, series_status[0]] = pos_result
    series_in = update_speed_and_accer(series_in, 2)
    return series_in


def wavefilter(data):
    # We will use the Daubechies(6) wavelet
    daubechies_num = 6
    wname = "db" + str(daubechies_num)
    datalength = data.shape[0]
    wavelet = pywt.Wavelet(wname)
    max_level = pywt.dwt_max_level(datalength, wavelet.dec_len)
    # print('maximun level is: %s' % max_level)
    # Initialize the container for the filtered data
    # Decompose the signal
    # coeff[0] is approximation coeffs, coeffs[1] is nth level detail coeff, coeff[-1] is first level detail coeffs
    coeffs = pywt.wavedec(data, wname, mode='smooth', level=max_level)
    # thresholding
    for j in range(max_level):
        coeffs[-j - 1] = np.zeros_like(coeffs[-j - 1])
    # Reconstruct the signal and save it
    filter_data = pywt.waverec(coeffs, wname, mode='smooth')
    fdata = filter_data[0:datalength]
    return fdata


def wavelet_filter(series_in, flag):
    filter_status = ['wavelet_filter_speed', 'wavelet_filter_cumu_dis', 'wavelet_filter_accer', 'wavelet_filter_jerk']
    if flag == 0:
        remove_outlier_speed_signal = series_in.loc[:, 'remove_outlier_speed_x'].to_numpy()
        # filter_status = ['wavelet_filter_speed_x', 'wavelet_filter_cumu_dis_x', 'wavelet_filter_accer_x', 'wavelet_filter_jerk_x']
    else:
        remove_outlier_speed_signal = series_in.loc[:, 'remove_outlier_speed_y'].to_numpy()
        # filter_status = ['wavelet_filter_speed_y', 'wavelet_filter_cumu_dis_y', 'wavelet_filter_accer_y', 'wavelet_filter_jerk_y']
    wavelet_filter_speed = wavefilter(remove_outlier_speed_signal)

    series_in.loc[:, filter_status[0]] = wavelet_filter_speed
    series_in.loc[:, filter_status[1]] = None
    series_in.loc[:, filter_status[2]] = None
    series_in.loc[:, filter_status[3]] = None
    for i in range(len(series_in['global_center_x'])):
        if i == 0:
            # start from the filtered value
            series_in.loc[i, filter_status[1]] = 0  # initial pos should be 0
        else:
            series_in.loc[i, filter_status[1]] = series_in.loc[i - 1, filter_status[1]] + (
                series_in.loc[i - 1, filter_status[0]] + series_in.loc[i, filter_status[0]]) * 0.5 * 0.04
    current_speed = filter_status[0]
    current_accer = filter_status[2]
    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_accer] = float(
                series_in.at[i + 2, current_speed] - series_in.at[i, current_speed]) / (float(0.08))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_accer] = float(
                series_in.at[i, current_speed] - series_in.at[i - 2, current_speed]) / (float(0.08))
        else:
            series_in.at[i, current_accer] = float(
                series_in.at[i + 1, current_speed] - series_in.at[i - 1, current_speed]) / (float(0.08))
    current_jerk = filter_status[3]
    for i in range(0, len(series_in['global_center_x'])):
        if i == 0:
            series_in.at[i, current_jerk] = float(
                series_in.at[i + 2, current_accer] - series_in.at[i, current_accer]) / (float(0.08))
        elif i == len(series_in['global_center_x']) - 1:
            series_in.at[i, current_jerk] = float(
                series_in.at[i, current_accer] - series_in.at[i - 2, current_accer]) / (float(0.08))
        else:
            series_in.at[i, current_jerk] = float(
                series_in.at[i + 1, current_accer] - series_in.at[i - 1, current_accer]) / (float(0.08))
    return series_in


def trajectory_correctness(coord_series_in, segment_id_in, veh_id_in, start_time_in, all_outlier_record):
    # this function remove outliers and filter the trajectory
    # input coord_series_in: ['global_center_x', 'global_center_y', 'cumu_dis', 'speed', 'accer']
    # output coord_series_in: ['global_center_x', 'global_center_y', 'cumu_dis', 'speed', 'accer', 'filter_cumu_dis', 'filter_speed', 'filter_accer']

    minimum_accer = -8
    maximum_accer = 5
    total_related_points = 50
    coord_series_in.reset_index(inplace=True, drop=True)
    # global all_outlier_record

    # remove outliers in acceleration, note that cubic spline interpolation is implemented on distance
    coord_series_in.loc[:, 'remove_outlier_cumu_dis_x'] = coord_series_in.loc[:, 'cumu_dis_x']
    coord_series_in.loc[:, 'remove_outlier_speed_x'] = coord_series_in.loc[:, 'speed_x']
    coord_series_in.loc[:, 'remove_outlier_accer_x'] = coord_series_in.loc[:, 'accer_x']
    coord_series_in.loc[:, 'remove_outlier_cumu_dis_y'] = coord_series_in.loc[:, 'cumu_dis_y']
    coord_series_in.loc[:, 'remove_outlier_speed_y'] = coord_series_in.loc[:, 'speed_y']
    coord_series_in.loc[:, 'remove_outlier_accer_y'] = coord_series_in.loc[:, 'accer_y']

    correctness_dir = ['remove_outlier_accer_x', 'remove_outlier_accer_y']

    for flag in range(2):
        outlier_label = 1
        while outlier_label:
            outlier_label = 0
            for m in range(len(coord_series_in['global_center_x'])):
                if coord_series_in.at[m, correctness_dir[flag]] >= maximum_accer or coord_series_in.at[m, correctness_dir[flag]] <= minimum_accer:
                    '''
                    print('Outlier info: Current segment: %s, vehicle id: %s, time: %s, position: %s' % (
                        segment_id_in, veh_id_in, round(m * 0.04 + start_time_in, 1), m))
                    '''
                    single_outlier_record = pd.DataFrame(np.zeros((1, 3)), columns=['segment_id', 'local_veh_id', 'outlier_time'])
                    single_outlier_record.loc[0, 'segment_id'] = segment_id_in
                    single_outlier_record.loc[0, 'local_veh_id'] = veh_id_in
                    single_outlier_record.loc[0, 'outlier_time'] = start_time_in + 0.04 * m
                    all_outlier_record.append(single_outlier_record)
                    first_point_pos = int(max(0, m - total_related_points / 2))
                    last_point_pos = int(min(len(coord_series_in.loc[:, correctness_dir[flag]]) - 1, m + total_related_points / 2))
                    if first_point_pos == 0:
                        last_point_pos = first_point_pos + total_related_points
                    if last_point_pos == len(coord_series_in.loc[:, correctness_dir[flag]]) - 1:
                        first_point_pos = last_point_pos - total_related_points
                    coord_series_in = optimization_based_outlier_removing(coord_series_in, first_point_pos, last_point_pos, minimum_accer,
                                                                          maximum_accer, flag)
                    if coord_series_in.empty:
                        return pd.DataFrame()
                    # plot_outlier_adjacent_trj(coord_series_in, m, first_point_pos, last_point_pos, segment_id_in, veh_id_in, start_time_in, 1, flag)
                    outlier_label = 0  # outlier still exsit in this loop
        coord_series_in = wavelet_filter(coord_series_in, flag)
        # 更新滤波后的轨迹结果
        if flag == 0:
            filter_status = ['filter_cumu_dis_x', 'filter_speed_x', 'filter_accer_x', 'filter_jerk_x']
        else:
            filter_status = ['filter_cumu_dis_y', 'filter_speed_y', 'filter_accer_y', 'filter_jerk_y']
        coord_series_in.loc[:, filter_status[0]] = coord_series_in.loc[:, 'wavelet_filter_cumu_dis'].to_numpy()
        coord_series_in.loc[:, filter_status[1]] = coord_series_in.loc[:, 'wavelet_filter_speed'].to_numpy()
        coord_series_in.loc[:, filter_status[2]] = coord_series_in.loc[:, 'wavelet_filter_accer'].to_numpy()
        coord_series_in.loc[:, filter_status[3]] = coord_series_in.loc[:, 'wavelet_filter_jerk'].to_numpy()
    return coord_series_in


def cumulated_dis_cal(coord_series_in, segment_id_in, veh_id_in, start_time_in, all_outlier_record):
    '''
    :param coord_series_in: -> ['global_center_x', 'global_center_y', 'speed_x', 'speed_y']

    '''
    # this function calculate the cumulated distance based on the given  global coordinates,
    # input coord_series_in: ['global_center_x', 'global_center_y', 'speed_x', 'speed_y']
    # output coord_series_in: ['global_center_x', 'global_center_y', 'speed_x', 'speed_y', 'cumu_dis', 'speed', 'accer', 'filter_cumu_dis',
    # 'filter_speed', 'filter_accer', 'speed_based_cumu_dis', 'speed_based_speed', 'speed_based_accer', 'speed_based_filter_cumu_dis',
    # 'speed_based_filter_speed', 'speed_based_accer']

    coord_series_in.reset_index(drop=True, inplace=True)

    coord_series_in.loc[:, 'cumu_dis_x'] = float(0)
    coord_series_in.loc[:, 'cumu_dis_y'] = float(0)
    coord_series_in.loc[:, 'speed_x'] = float(0)
    coord_series_in.loc[:, 'speed_y'] = float(0)
    coord_series_in.loc[:, 'accer_x'] = float(0)
    coord_series_in.loc[:, 'accer_y'] = float(0)
    coord_series_in.loc[:, 'jerk_x'] = float(0)
    coord_series_in.loc[:, 'jerk_y'] = float(0)
    coord_series_in.loc[:, 'speed_based_cumu_dis_x'] = float(0)
    coord_series_in.loc[:, 'speed_based_cumu_dis_y'] = float(0)
    coord_series_in.loc[:, 'speed_based_speed_x'] = float(0)
    coord_series_in.loc[:, 'speed_based_speed_y'] = float(0)
    coord_series_in.loc[:, 'speed_based_accer_x'] = float(0)
    coord_series_in.loc[:, 'speed_based_accer_y'] = float(0)
    coord_series_in.loc[:, 'speed_based_jerk_x'] = float(0)
    coord_series_in.loc[:, 'speed_based_jerk_y'] = float(0)

    for i in range(1, len(coord_series_in['global_center_x'])):
        pre_x = coord_series_in['global_center_x'].iloc[i - 1]
        pre_y = coord_series_in['global_center_y'].iloc[i - 1]
        post_x = coord_series_in['global_center_x'].iloc[i]
        post_y = coord_series_in['global_center_y'].iloc[i]
        single_dis_x = post_x - pre_x
        single_dis_y = post_y - pre_y
        coord_series_in.loc[i, 'cumu_dis_x'] = coord_series_in.loc[i - 1, 'cumu_dis_x'] + single_dis_x  # 计算累计距离
        coord_series_in.loc[i, 'cumu_dis_y'] = coord_series_in.loc[i - 1, 'cumu_dis_y'] + single_dis_y

    coord_series_in = update_speed_and_accer(coord_series_in, 0)
    coord_series_in = speed_based_update_distance_and_accer(coord_series_in)

    coord_series_in.loc[:, 'filter_cumu_dis_x'] = coord_series_in.loc[:, 'cumu_dis_x'].to_numpy()
    coord_series_in.loc[:, 'filter_speed_x'] = coord_series_in.loc[:, 'speed_x'].to_numpy()
    coord_series_in.loc[:, 'filter_accer_x'] = coord_series_in.loc[:, 'accer_x'].to_numpy()
    coord_series_in.loc[:, 'filter_jerk_x'] = 0
    coord_series_in.loc[:, 'filter_cumu_dis_y'] = coord_series_in.loc[:, 'cumu_dis_y'].to_numpy()
    coord_series_in.loc[:, 'filter_speed_y'] = coord_series_in.loc[:, 'speed_y'].to_numpy()
    coord_series_in.loc[:, 'filter_accer_y'] = coord_series_in.loc[:, 'accer_y'].to_numpy()
    coord_series_in.loc[:, 'filter_jerk_y'] = 0
    coord_series_in = trajectory_correctness(coord_series_in, segment_id_in, veh_id_in, start_time_in, all_outlier_record)
    return coord_series_in


def pair_cf_coord_cal(leader_id, leader_trj_in, follower_id, follower_trj_in, av_label, all_outlier_record):
    # convert 2-d coordinates to 1-d longitudinal coordinates
    # note that the leader and follower interacts with each other
    # av_label is to determine whether av is leader or follower (0 for follower, 1 for leader, 2 for non-av pair)
    all_seg_paired_cf_trj_final = pd.DataFrame()
    # all_seg_paired_cf_trj_with_comparison = pd.DataFrame()
    min_local_time = max(leader_trj_in['local_time_stamp'].min(), follower_trj_in['local_time_stamp'].min())
    max_local_time = min(leader_trj_in['local_time_stamp'].max(), follower_trj_in['local_time_stamp'].max())
    leader_trj_in = leader_trj_in.loc[leader_trj_in['local_time_stamp'] >= min_local_time, :]
    leader_trj_in = leader_trj_in.loc[leader_trj_in['local_time_stamp'] <= max_local_time, :]
    follower_trj_in = follower_trj_in.loc[follower_trj_in['local_time_stamp'] >= min_local_time, :]
    follower_trj_in = follower_trj_in.loc[follower_trj_in['local_time_stamp'] <= max_local_time, :]
    leader_trj_in = leader_trj_in.sort_values(['local_time_stamp'])
    follower_trj_in = follower_trj_in.sort_values(['local_time_stamp'])
    out_leader_trj = pd.DataFrame(leader_trj_in[['segment_id', 'veh_id', 'length', 'local_time_stamp']].to_numpy(),
                                  columns=['segment_id', 'local_veh_id', 'length', 'local_time'])
    out_leader_trj.loc[:, 'follower_id'] = follower_id
    out_leader_trj.loc[:, 'leader_id'] = leader_id
    out_follower_trj = pd.DataFrame(follower_trj_in[['segment_id', 'veh_id', 'length', 'local_time_stamp']].to_numpy(),
                                    columns=['segment_id', 'local_veh_id', 'length', 'local_time'])
    out_follower_trj.loc[:, 'follower_id'] = follower_id
    out_follower_trj.loc[:, 'leader_id'] = leader_id
    # cf_paired_trj_plot(out_leader_trj, out_follower_trj, av_label)
    temp_current_segment_id = out_follower_trj['segment_id'].iloc[0]  # cutin id
    temp_start_time = out_follower_trj['local_time'].iloc[0]
    leader_cumu_dis = cumulated_dis_cal(
        leader_trj_in.loc[:, ['global_center_x', 'global_center_y', 'speed_x', 'speed_y']], temp_current_segment_id, leader_id, temp_start_time, all_outlier_record)
    follower_cumu_dis = cumulated_dis_cal(
        follower_trj_in.loc[:, ['global_center_x', 'global_center_y', 'speed_x', 'speed_y']], temp_current_segment_id, follower_id, temp_start_time, all_outlier_record)
    if leader_cumu_dis.empty or follower_cumu_dis.empty:
        print('Scenario %s has no feasible solution' % str(temp_current_segment_id))
        return pd.DataFrame()
    pre_x_1 = leader_trj_in['global_center_x'].iloc[0]
    pre_y_1 = leader_trj_in['global_center_y'].iloc[0]
    post_x_1 = follower_trj_in['global_center_x'].iloc[0]
    post_y_1 = follower_trj_in['global_center_y'].iloc[0]
    initial_dis = [post_x_1 - pre_x_1, post_y_1 - pre_y_1]
    tags = ['_x', '_y']
    for i in range(2):
        out_follower_trj.loc[:, 'position' + tags[i]] = follower_cumu_dis['cumu_dis' + tags[i]].to_numpy() + initial_dis[i]
        out_follower_trj.loc[:, 'remove_outlier_pos' + tags[i]] = follower_cumu_dis['remove_outlier_cumu_dis' + tags[i]].to_numpy() + initial_dis[i]
        out_follower_trj.loc[:, 'filter_pos' + tags[i]] = follower_cumu_dis['filter_cumu_dis' + tags[i]].to_numpy() + initial_dis[i]
        # out_follower_trj.loc[:, 'wavelet_filter_pos' + tags[i]] = follower_cumu_dis['wavelet_filter_cumu_dis' + tags[i]].to_numpy()
        out_follower_trj.loc[:, 'speed' + tags[i]] = follower_cumu_dis['speed' + tags[i]].to_numpy()
        out_follower_trj.loc[:, 'remove_outlier_speed' + tags[i]] = follower_cumu_dis['remove_outlier_speed' + tags[i]].to_numpy()
        out_follower_trj.loc[:, 'filter_speed' + tags[i]] = follower_cumu_dis['filter_speed' + tags[i]].to_numpy()
        # out_follower_trj.loc[:, 'wavelet_filter_speed' + tags[i]] = follower_cumu_dis['wavelet_filter_speed' + tags[i]].to_numpy()
        out_follower_trj.loc[:, 'accer' + tags[i]] = follower_cumu_dis['accer' + tags[i]].to_numpy()
        out_follower_trj.loc[:, 'remove_outlier_accer' + tags[i]] = follower_cumu_dis['remove_outlier_accer' + tags[i]].to_numpy()
        out_follower_trj.loc[:, 'filter_accer' + tags[i]] = follower_cumu_dis['filter_accer' + tags[i]].to_numpy()
        # out_follower_trj.loc[:, 'wavelet_filter_accer' + tags[i]] = follower_cumu_dis['wavelet_filter_accer' + tags[i]].to_numpy()
        out_follower_trj.loc[:, 'jerk' + tags[i]] = follower_cumu_dis['jerk' + tags[i]].to_numpy()
        out_follower_trj.loc[:, 'filter_jerk' + tags[i]] = follower_cumu_dis['filter_jerk' + tags[i]].to_numpy()
        # out_follower_trj.loc[:, 'wavelet_filter_jerk' + tags[i]] = follower_cumu_dis['wavelet_filter_jerk' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'position' + tags[i]] = leader_cumu_dis['cumu_dis' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'remove_outlier_pos' + tags[i]] = leader_cumu_dis['remove_outlier_cumu_dis' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'filter_pos' + tags[i]] = leader_cumu_dis['filter_cumu_dis' + tags[i]].to_numpy()
        # out_leader_trj.loc[:, 'wavelet_filter_pos' + tags[i]] = leader_cumu_dis['wavelet_filter_cumu_dis' + tags[i]].to_numpy() + initial_dis[i]
        out_leader_trj.loc[:, 'speed' + tags[i]] = leader_cumu_dis['speed' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'remove_outlier_speed' + tags[i]] = leader_cumu_dis['remove_outlier_speed' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'filter_speed' + tags[i]] = leader_cumu_dis['filter_speed' + tags[i]].to_numpy()
        # out_leader_trj.loc[:, 'wavelet_filter_speed' + tags[i]] = leader_cumu_dis['wavelet_filter_speed' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'accer' + tags[i]] = leader_cumu_dis['accer' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'remove_outlier_accer' + tags[i]] = leader_cumu_dis['remove_outlier_accer' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'filter_accer' + tags[i]] = leader_cumu_dis['filter_accer' + tags[i]].to_numpy()
        # out_leader_trj.loc[:, 'wavelet_filter_accer' + tags[i]] = leader_cumu_dis['wavelet_filter_accer' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'jerk' + tags[i]] = leader_cumu_dis['jerk' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'filter_jerk' + tags[i]] = leader_cumu_dis['filter_jerk' + tags[i]].to_numpy()
        # out_leader_trj.loc[:, 'wavelet_filter_jerk' + tags[i]] = leader_cumu_dis['wavelet_filter_jerk' + tags[i]].to_numpy()
        # ======基于速度的轨迹数据======
        out_follower_trj.loc[:, 'speed_based_position' + tags[i]] = follower_cumu_dis['speed_based_cumu_dis' + tags[i]].to_numpy() + initial_dis[i]
        out_follower_trj.loc[:, 'speed_based_speed' + tags[i]] = follower_cumu_dis['speed_based_speed' + tags[i]].to_numpy()
        out_follower_trj.loc[:, 'speed_based_accer' + tags[i]] = follower_cumu_dis['speed_based_accer' + tags[i]].to_numpy()
        out_follower_trj.loc[:, 'speed_based_jerk' + tags[i]] = follower_cumu_dis['speed_based_jerk' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'speed_based_position' + tags[i]] = leader_cumu_dis['speed_based_cumu_dis' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'speed_based_speed' + tags[i]] = leader_cumu_dis['speed_based_speed' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'speed_based_accer' + tags[i]] = leader_cumu_dis['speed_based_accer' + tags[i]].to_numpy()
        out_leader_trj.loc[:, 'speed_based_jerk' + tags[i]] = leader_cumu_dis['speed_based_jerk' + tags[i]].to_numpy()
    # plot speed and acc figure
    '''
    before_and_after_remove_outlier_plot(out_follower_trj)
    before_and_after_remove_outlier_plot(out_leader_trj)
    before_and_after_filtering_plot(out_follower_trj)
    before_and_after_filtering_plot(out_leader_trj)
    '''
    # save cf paired trj
    # all_seg_paired_cf_trj = pd.concat([all_seg_paired_cf_trj, pd.concat([out_leader_trj, out_follower_trj])])
    # all_seg_paired_cf_trj_with_comparison = pd.concat([out_leader_trj, out_follower_trj])
    out_follower_trj_final = out_follower_trj.loc[
        :, ['segment_id', 'local_veh_id', 'length', 'local_time', 'follower_id', 'leader_id',
            'filter_pos_x', 'filter_speed_x', 'filter_accer_x', 'filter_pos_y', 'filter_speed_y', 'filter_accer_y']]
    out_follower_trj_final.columns = [
        'segment_id', 'local_veh_id', 'length', 'local_time', 'follower_id', 'leader_id',
        'filter_pos_x', 'filter_speed_x', 'filter_accer_x', 'filter_pos_y', 'filter_speed_y', 'filter_accer_y']
    out_leader_trj_final = out_leader_trj.loc[
        :, ['segment_id', 'local_veh_id', 'length', 'local_time', 'follower_id', 'leader_id',
            'filter_pos_x', 'filter_speed_x', 'filter_accer_x', 'filter_pos_y', 'filter_speed_y', 'filter_accer_y']]
    out_leader_trj_final.columns = [
        'segment_id', 'local_veh_id', 'length', 'local_time', 'follower_id', 'leader_id',
        'filter_pos_x', 'filter_speed_x', 'filter_accer_x', 'filter_pos_y', 'filter_speed_y', 'filter_accer_y']
    all_seg_paired_cf_trj_final = pd.concat([out_leader_trj_final, out_follower_trj_final])
    # plot the car following trj of both follower and leader
    '''
    cf_paired_trj_plot(out_leader_trj_final, out_follower_trj_final, av_label)
    '''
    return all_seg_paired_cf_trj_final


def csv_plot(oldcsv_path, newcsv_path, ego_id, cutin_id, scene_id):
    def getPos(csv_path, ego_id, cutin_id):
        df = pd.read_csv(csv_path)
        df_ego = df[df['id'] == ego_id]
        df_cutin = df[df['id'] == cutin_id]
        x1, y1 = df_ego['x'].tolist(), df_ego['y'].tolist()
        x2, y2 = df_cutin['x'].tolist(), df_cutin['y'].tolist()
        min_x = min(min(x1), min(x2)) - 10
        max_x = max(max(x1), max(x2)) + 10
        min_y = min(min(y1), min(y2)) - 10
        max_y = max(max(y1), max(y2)) + 10
        return x1, y1, x2, y2, min_x, max_x, min_y, max_y

    oldPos = getPos(oldcsv_path, ego_id, cutin_id)
    newPos = getPos(newcsv_path, ego_id, cutin_id)

    plt.ion()
    for i in range(len(newPos[0])):
        plt.subplot(2, 1, 1)
        plt.xlim(oldPos[4], oldPos[5])
        plt.ylim(oldPos[6], oldPos[7])
        plt.scatter(oldPos[0][i], oldPos[1][i], c='r')
        plt.scatter(oldPos[2][i], oldPos[3][i], c='b')
        plt.title('Scenario ' + str(scene_id) + ' GIF' + ' Before and After Data Processing')
        plt.subplot(2, 1, 2)
        plt.xlim(newPos[4], newPos[5])
        plt.ylim(newPos[6], newPos[7])
        plt.scatter(newPos[0][i], newPos[1][i], c='r')
        plt.scatter(newPos[2][i], newPos[3][i], c='b')
        plt.pause(1e-7)
        plt.clf()
    plt.ioff()


def directionUnify(all_id, df_scene):
    ego_id = all_id[0]
    df_ego = df_scene[df_scene['veh_id'] == ego_id]
    if df_ego.loc[0, 'global_center_x'] > df_ego.loc[len(df_ego) - 1, 'global_center_x']:
        df_scene.loc[:, 'global_center_x'] = df_scene.loc[:, 'global_center_x'] * -1
        df_scene.loc[:, 'global_center_y'] = df_scene.loc[:, 'global_center_y'] * -1
        df_scene.loc[:, 'speed_x'] = df_scene.loc[:, 'speed_x'] * -1
        df_scene.loc[:, 'speed_y'] = df_scene.loc[:, 'speed_y'] * -1
    return


if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)

    scene_rootpath = r"D:\VAE_sample\cutin\std\scenarios"
    scenario_path = []
    for filepath, dirnames, filenames in os.walk(scene_rootpath):
        for dirname in dirnames:
            dir_path = os.path.join(filepath, dirname)
            for filepath1, dirnames1, filenames1 in os.walk(dir_path):
                for filename in filenames1:
                    if '_3cars.csv' in filename and 'processed.csv' not in filename:
                        scenario_path.append(os.path.join(filepath1, filename))
                    if '3cars_processed' in filename:
                        scenario_path.pop()
    noSolution_count = 0
    completion = 0
    pbar_iteration = tqdm(total=len(scenario_path), desc='[scenarios]')
    for i in range(len(scenario_path)):
        flag = 0
        scene_path = scenario_path[i]
        df_scene = pd.read_csv(scene_path)
        scene_id = scene_path.split('\\')[-1][:4]
        df_scene['segment_id'] = [scene_id] * df_scene.shape[0]
        df_scene.rename(columns={
            'frame': 'local_time_stamp', 'id': 'veh_id', 'width': 'length', 'x': 'global_center_x', 'y': 'global_center_y',
            'xVelocity': 'speed_x', 'yVelocity': 'speed_y', 'precedingId': 'precedingId'}, inplace=True)
        ego_id = df_scene.loc[0, 'veh_id']
        all_id = df_scene['veh_id'].unique()
        directionUnify(all_id, df_scene)
        df_ego = df_scene[df_scene['veh_id'] == ego_id]
        init_pos = [df_ego['global_center_x'].iloc[0], df_ego['global_center_y'].iloc[0]]
        df_final = pd.DataFrame()
        for nonego_id in all_id:
            if nonego_id != ego_id:
                df_nonego = df_scene[df_scene['veh_id'] == nonego_id]
                all_outlier_record = pd.DataFrame()
                out_trj = pd.DataFrame()
                out_trj = pair_cf_coord_cal(ego_id, df_ego, nonego_id, df_nonego, 0, all_outlier_record)
                if not out_trj.empty:
                    df_output = out_trj.loc[:, ['local_veh_id', 'length', 'local_time', 'filter_pos_x', 'filter_speed_x',
                                            'filter_accer_x', 'filter_pos_y', 'filter_speed_y', 'filter_accer_y']]
                    df_output.rename(columns={
                        'local_veh_id': 'id', 'length': 'width', 'local_time': 'frame', 'filter_pos_x': 'x', 'filter_pos_y': 'y', 'filter_speed_x': 'xVelocity',
                        'filter_speed_y': 'yVelocity', 'filter_accer_x': 'xAcceleration', 'filter_accer_y': 'yAcceleration'}, inplace=True)
                    # df_output['x'] = df_output['x'].map(lambda x: x + init_pos[0])
                    # df_output['y'] = df_output['y'].map(lambda x: x + init_pos[1])
                    if df_final.empty:
                        df_final = df_output.copy()
                    else:
                        df_merge = df_output[df_output['id'] == nonego_id]
                        df_final = pd.concat([df_final, df_merge])
                else:
                    flag = 1
                    continue
        if flag:
            noSolution_count += 1
            pbar_iteration.update(1)
            continue
        output_path = scene_path.split('.')[0] + '_processed.csv'
        df_final.to_csv(output_path, index=None)
        completion += 1
        pbar_iteration.update(1)
    pbar_iteration.write('[*] Finish Data Processing')
    pbar_iteration.close()
    print('无解场景数：{}, 完成增强场景数：{}'.format(str(noSolution_count), str(completion)))
