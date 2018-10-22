%% Load data, remove duplicate rows and sort ascending

clear; clc; close all;

% load data
% raw_trips should contain each (vehicle, timestamp) pair in a separate mat
% file
load('../data/raw_trips_191018.mat');

% Iterate over variables in  workspace
A = whos;

fprintf('Loading variables...\n');

c = {};
for ii = 1:numel(A)

    % Current timetable
    tt_name = A(ii).name;
    temp = split(tt_name, '_');
    
    fprintf('%s...', tt_name);
    
    feature_id = temp{1};
    veh_id = temp{2};
    
    TT = eval(tt_name);
    
    % Remove duplicates and sort to ascending order
    TT = retime(TT, unique(TT.timestamp));
    TT = sortrows(TT, 'timestamp', 'ascend');
    assignin('base', tt_name, TT);
    fprintf('OK!\n', tt_name);
    
    % --Write meta
    
    % feature name
    c{ii, 1} = feature_id;
    % vehicle id
    c{ii, 2} = veh_id;
    % Number of datapoints in subset
    c{ii, 3} = height(TT);
    % Mean duration between consecutive timesteps
    c{ii, 4} = mean(seconds(diff(TT.timestamp)));
    % median duration between consecutive timesteps
    c{ii, 5} = median(seconds(diff(TT.timestamp)));
    % data payload
    c{ii, 6} = TT;

end

% clean up
clearvars -except c
c = cell2table(c);
c.Properties.VariableNames = {'feature', 'vehicle', 'n_datapoints',...
    'mean_diff', 'median_diff', 'payload'};



%% Interpolation

% esim. {'axleweight', 'gpsalt', 'gpslat', 'gpslong', 'soc', 'speed', 'temp'}
features = unique(c(:,1));

% esim. {'133353', '174460', '186973', 192493'}
vehicles = unique(c(:,2));

% load interpolation table
load('latlong_interp.mat');

fprintf('\nInterpolating values...\n');

% Iterate over vehicles and build the initial datasets by interpolation
for i = 1:height(vehicles)
    veh = vehicles{i, 1}{1};
    
    % grab the relevant subsets for current vehicle
    gpslat = c{strcmp(c.feature, 'gpslat') & strcmp(c.vehicle, veh), 'payload'}{1};
    gpslong = c{strcmp(c.feature, 'gpslong') & strcmp(c.vehicle, veh), 'payload'}{1};
    soc = c{strcmp(c.feature, 'soc') & strcmp(c.vehicle, veh), 'payload'}{1};
    speed = c{strcmp(c.feature, 'speed') & strcmp(c.vehicle, veh), 'payload'}{1};
    tempamb = c{strcmp(c.feature, 'tempamb') & strcmp(c.vehicle, veh), 'payload'}{1};
    tempcabin = c{strcmp(c.feature, 'tempcabin') & strcmp(c.vehicle, veh), 'payload'}{1};    
    axleweight = c{strcmp(c.feature, 'axleweight') & strcmp(c.vehicle, veh), 'payload'}{1};
    odoint = c{strcmp(c.feature, 'odo') & strcmp(c.vehicle, veh), 'payload'}{1};
    ododec = c{strcmp(c.feature, 'ododec') & strcmp(c.vehicle, veh), 'payload'}{1};
    braking = c{strcmp(c.feature, 'braking') & strcmp(c.vehicle, veh), 'payload'}{1};
    
    % combine odomoeter integer and decimal parts
    odo = synchronize(odoint, ododec, 'last', 'previous');
    odo.value = odo.value_ododec + odo.value_odoint;
    odo.value_ododec = [];
    odo.value_odoint = [];
    
    
    % include exact matches only
    data = synchronize(gpslong, gpslat, 'intersection');
    % interpolate altitude
    data.altitude = latlong_interp([data.value_gpslat data.value_gpslong]);
    % include all from speed, use linear interpolation for lat/long/alt
    data = synchronize(data, speed, 'last', 'nearest');
    % include all from data, use linear interpolation for odo
    data = synchronize(data, odo, 'first', 'linear');
    % include all from data, use linear interp for soc
    data = synchronize(data, soc, 'first', 'linear');
    % include all from data, take nearest neighbor for ambient temp
    data = synchronize(data, tempamb, 'first', 'nearest');
    % include all from data, take nearest neighbor for cabin temp
    data = synchronize(data, tempcabin, 'first', 'nearest');
    % include all from data, take nearest neighbor for axleweight
    data = synchronize(data, axleweight, 'first', 'nearest');
    % include all from data, take previous value for odo

    
    data.vehicle(:) = string(veh);
        
    data.Properties.VariableNames = {...
        'gpslong', 'gpslat', 'gpsalt', 'speed', 'odo'... 
        'soc', 'temp_amb', 'temp_cabin' 'axleweight', 'vehicle'};
    
    marketplace_lat = 60.45093;
    marketplace_long = 22.26617;
    marketplace2_lat = 60.45082;
    marketplace2_long = 22.26717;
    harbour_lat = 60.43491;
    harbour_long = 22.21961;
    airport_lat = 60.51109;
    airport_long = 22.27421;
    
    data.dist_to_airport = lldistkm([data.gpslat data.gpslong], [airport_lat airport_long]);
    data.dist_to_market = lldistkm([data.gpslat data.gpslong], [marketplace_lat marketplace_long]);
    data.dist_to_harbour = lldistkm([data.gpslat data.gpslong], [harbour_lat harbour_long]);
    data.dist_to_market2 = lldistkm([data.gpslat data.gpslong], [marketplace2_lat marketplace2_long]);
    data.dist_to_nearest_waypoint = min(min(min(data.dist_to_harbour, data.dist_to_airport), data.dist_to_market), data.dist_to_market2);

    % set threshold for sequence clipping
    threshold = 0.025;
    %data = data(data.dist_to_nearest_waypoint >= threshold, :);
    
    assignin('base', join(['data', veh], '_'), data);
    fprintf('Wrote interpolation output to %s\n', veh);

end

data = [data133353; data174460; data186973; data192493];

%writetable(timetable2table(data(:, 1:9)), 'summary_stats_prefilter_2.csv');

% clean up
clearvars -except c data


%% Pre-outlier visualizations

close all;

figure;
histogram2(data.gpslong,data.gpslat, 100);
title('GPSLat, GPSLong - filtered')
% conclusion: 

figure;
histogram(data.gpsalt, 50);
set(gca, 'YScale', 'log');
set(gcf, 'Position', [100 100 560 130])
title('GPSalt - filtered');
% conclusion: 

figure;
histogram(data.speed, 50)
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('Vehicle Speed - filtered')
% conclusion: 

figure;
histogram(data.odo, 50)
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('Odometer - filtered')
% conclusion: 

figure;
histogram(data.soc, 50);
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('SOC - filtered')
% conclusion: 
figure;
histogram(data.temp_amb, 50);
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('Ambient temp - filtered')
% conclusion:

figure;
histogram(data.temp_cabin, 50);
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('Cabin temp - filtered')
% conclusion:

figure;
histogram(data.axleweight, 50);
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('Axle Weight - filtered')
% conclusion:


%% Process outliers

%not driving in turku area
data = data(data.dist_to_nearest_waypoint < 50, :);

%% Visualize

figure;
histogram2(data.gpslong,data.gpslat, 100);
title('GPSLat, GPSLong - filtered')

% conclusion: 
figure;
histogram(data.gpsalt, 50);
set(gca, 'YScale', 'log');
set(gcf, 'Position', [100 100 560 130])
title('GPSalt - filtered');
% conclusion: 

figure;
histogram(data.speed, 50)
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('Vehicle Speed - filtered')
% conclusion: 

figure;
histogram(data.odo, 50)
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('Odometer - filtered')
% conclusion: 

figure;
histogram(data.soc, 50);
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('SOC - filtered')
% conclusion: 
figure;
histogram(data.temp_amb, 50);
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('Ambient temp - filtered')
% conclusion:

figure;
histogram(data.temp_cabin, 50);
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('Cabin temp - filtered')
% conclusion:

figure;
histogram(data.axleweight, 50);
set(gca, 'YScale', 'log')
set(gcf, 'Position', [100 100 560 130])
title('Axle Weight - filtered')
% conclusion:

%% Process remaining outliers

speed = data.speed;
soc = data.soc;
temp = data.temp;
axleweight = data.axleweight;

[data.speed, speed_isoutlier, speed_l, speed_u, ~] = ...
    filloutliers(speed, 'linear', 'quartile', 'ThresholdFactor', 2);

[data.temp, temp_isoutlier, temp_l, temp_u, ~] = ...
    filloutliers(temp, 'linear', 'quartile', 'ThresholdFactor', 2);

[data.axleweight, axlw_isoutlier, axlw_l, axlw_u, ~] = ...
    filloutliers(axleweight, 'linear', 'quartile', 'ThresholdFactor', 2);

clear speed soc temp axleweight;

%% Post outlier removal visualizations

figure;
histogram2(data.gpslong,data.gpslat, 2500);
title('GPSLat, GPSLong')
% conclusion: 

figure;
histogram(data.gpsalt, 50);
set(gca, 'YScale', 'log');
title('GPSalt');
% conclusion: 

figure;
histogram(data.soc, 50);
set(gca, 'YScale', 'log')
title('SOC - filtered')
% conclusion: 

figure;
subplot(3, 1, 1);
histogram(data.speed, 50)
set(gca, 'YScale', 'log')
title('Vehicle Speed')
% conclusion: 

subplot(3, 1, 2);
histogram(data.temp, 50);
set(gca, 'YScale', 'log')
title('Ambient temp')
% conclusion: 

subplot(3, 1, 3);
histogram(data.axleweight, 50);
set(gca, 'YScale', 'log')
title('Axle Weight')
% conclusion:

%% Make sequences

diffs = [0; seconds(diff(data.timestamp))];
window_size = 24;
diff_threshold = 30;

h = waitbar(0, 'indexing...');
seqid = 1;
i = 1;
while true
    
    % stopping condition
    if i >= height(data)
        break
    end
    
    % valid sequence
    % threshold timestep hop is not exceeded
    % all data is from same vehicle
    if max(diffs(i:i+window_size-1)) <= diff_threshold && length(unique(data.vehicle(i:i+window_size-1, :))) == 1

        data.sequence(i:i+window_size-1, :) = seqid;
        seqid = seqid+1;
        i = i+window_size;
    else
        i = i+1;
    end
    waitbar(i / height(data), h, ['Building indices, ', int2str(i), ' / ', int2str(height(data)), ' samples processed'])
end

percent_used = length(data.sequence(data.sequence ~= 0)) / height(data) * 100;
fprintf('Sequencing complete, data loss is %f percent\n', percent_used) 
fprintf('Produced %i non-overlapping sequences\n', seqid - 1)

