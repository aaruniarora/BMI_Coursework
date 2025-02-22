clc
clear all

load('monkeydata_training.mat');

%%

reach_angles = [1/6,7/18,11/18,15/18,19/18,23/18,31/18,35/18].*pi;

angle_no = length(reach_angles);

time_step = 1*10^-3;

trial1 = trial(1,1);

trials = 100;

spikes = trial1.spikes;
handpos = trial1.handPos;

before = 300;
after =100;

spikes = spikes(:, 301:end-100);

shape = size(spikes);

neurons = shape(1);
time_points = shape(2);

time = linspace (0,time_points,time_points);
%%
figure;
hold on
for i = 1:neurons
plot(time,spikes(i,:))
end
hold off

%%
% Sum across units (rows) to get total activity per time point

spike_counts = sum(spikes, 1);  % 1×672 vector

figure;
bar(spike_counts);
xlabel('Time Bins');
ylabel('Spike Count');
title('Spike Frequency Over Time');
grid on;

%%
clc
spikes_all = {};
for i = 1:trials
    for x = 1:angle_no
        array = trial(i,x).spikes;
        spikes_all{i,x} = array;
    end
end

handpos_all = {};
for i = 1:trials
    for x = 1:angle_no
        array = trial(i,x).handPos;
        handpos_all{i,x} = array;
    end
end

