function [meanRMSE, meanAccuracy] = testFunction_for_students_MTb_kfold( ...
                                   teamName, k, use_rng )
% -------------------------------------------------------------------------
% k‑fold CV across 5 random seeds × 5 regression methods.
%   (1) box‑plot: RMSE  vs regression method
%   (2) box‑plot: time  vs regression method
%
% METHODS: standard | ridge | lasso | poly | LMS
%
% EXAMPLE CALL
%   RMSE = testFunction_for_students_MTb_kfold('report2', 2, true);
% -------------------------------------------------------------------------

    if nargin < 2, k       = 10;  end           % default 10‑fold
    if nargin < 3, use_rng = false; end         % rng optional

    %% ---------------- CONFIGURATION ------------------------------------
    methods  = {'standard','ridge','lasso','poly','LMS','kalman'};   % <- added LMS
    % ---> create display‑only labels
    pretty = {'PCR', 'Ridge','Lasso','PR','LMS','KF'};
    rngSeeds  = [401,501,601,701,801];                       % 5 seeds
    nM = numel(methods);
    nS = numel(rngSeeds);

    %% ---------------- PRE‑ALLOC RESULTS --------------------------------
    RMSE = nan(nM,nS);
    Time = nan(nM,nS);
    Acc  = nan(nM,nS);

    %% ---------------- DATA LOAD & PATH ---------------------------------
    load('/Users/mahadparwaiz/Desktop/Imperial College London/BMI/BMI-Competition_Mywork/monkeydata_training.mat','trial');
    addpath(genpath(teamName));          % estimator folders

    %% ---------------- MAIN GRID ----------------------------------------
    [nTrials, ~] = size(trial);          % 100 × 8

    for m = 1:nM
        method = methods{m};

        % pick the correct training / estimator pair ---------------------
        switch method
            case {'standard','ridge','lasso','poly'}
                trainFcn = @positionEstimatorTraining;
                estFcn   = @positionEstimator;
            case 'LMS'
                trainFcn = @LMS_positionEstimatorTraining;
                estFcn   = @LMS_positionEstimator;
            case 'kalman'
                trainFcn = @kalman_positionEstimatorTraining;
                estFcn   = @kalman_positionEstimator;
            otherwise
                error('Unknown method string.');
        end

        for s = 1:nS
            if use_rng, rng(rngSeeds(s)); end

            foldID   = crossvalind('Kfold', nTrials, k);
            foldRMSE = zeros(1,k);
            foldAcc  = zeros(1,k);
            foldTimes  = zeros(1,k);          % <‑‑ NEW  (one entry per fold)

            for f = 1:k
                trainIdx     =  foldID ~= f;
                testIdx      = ~trainIdx;
                trainingData = trial(trainIdx , :);
                testData     = trial(testIdx  , :);
                tStart = tic;     

                % ---- train -------------------------------------------------
                if any(strcmp(method, {'LMS','kalman'}))
                    model = trainFcn(trainingData);                 % no arg
                else
                    model = trainFcn(trainingData, method);         % pass arg
                end

                % ---- evaluate ----------------------------------------------
                [foldRMSE(f), foldAcc(f)] = ...
                        evaluate_fold(testData, model, estFcn);
                foldTimes(f) = toc(tStart);
            end

            Time(m,s) = mean(foldTimes); 
            RMSE(m,s) = mean(foldRMSE);
            Acc(m,s)  = mean(foldAcc);

            fprintf('Method %-8s | seed %d | RMSE %.4f | acc %.2f%% | %.1fs\n', ...
                    method, rngSeeds(s), RMSE(m,s), 100*Acc(m,s), Time(m,s));
        end
    end

    rmpath(genpath(teamName));
%% --------------------------- BOX‑PLOTS ---------------------------------
%  Creates two publication‑ready figures:
%     • fig1 – RMSE vs. method
%     • fig2 – Avg. runtime vs. method
%  Each figure
%     – has 45° x‑tick labels
%     – shows the median value as bold text on a white patch (so it never
%       blends into the tiny box)
%     – carries a horizontal legend beneath the axes that repeats the same
%       median values (comment‑out the legend block if you prefer not to
%       duplicate the numbers).

% ---------- gather per‑method medians -----------------------------------
medRMSE = median(RMSE ,2);
medTime = median(Time ,2);

% ---------- common cosmetics -------------------------------------------
xtickRotation = 45;     % 45° slanted method labels
w = 6;  hFig = 4;       % final PDF size (inches)

% ========================================================================
% 1️⃣  RMSE
% ========================================================================
fig1 = figure;
boxH = boxplot(RMSE','Labels',pretty);
set(boxH,'LineWidth',2);

ax = gca;
ax.FontSize   = 14;
ax.FontWeight = 'bold';
ax.LineWidth  = 1.5;
xlabel('Method','FontSize',16,'FontWeight','bold');
ylabel('RMSE (cm)','FontSize',16,'FontWeight','bold');
xtickangle(ax,xtickRotation);

% ---- text label with white background ----------------------------------
hold on
offset = 0.05*range(ylim(ax));         % 5 % above the median line
for i = 1:nM
    text(i, medRMSE(i)+offset, sprintf('%.3f',medRMSE(i)), ...
        'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
        'FontSize',12, 'FontWeight','bold', ...
        'BackgroundColor','w', 'Margin',0.1);
end

hold off

% ---- figure sizing + export -------------------------------------------
fig1.Units = 'inches';
fig1.Position = [1 1 w hFig];
fig1.PaperUnits = 'inches';
fig1.PaperPosition = [0 0 w hFig];
fig1.PaperSize = [w hFig];
fig1.PaperPositionMode = 'manual';
exportgraphics(fig1,'RMSE_boxplot.pdf', ...
                   'ContentType','vector', ...
                   'BackgroundColor','none');

% ========================================================================
% 2️⃣  Average runtime
% ========================================================================
fig2 = figure;
boxH2 = boxplot(Time','Labels',pretty);
set(boxH2,'LineWidth',2);

ax = gca;
ax.FontSize   = 14;
ax.FontWeight = 'bold';
ax.LineWidth  = 1.5;
xlabel('Method','FontSize',16,'FontWeight','bold');
ylabel('Avg Time Per Run (s)','FontSize',16,'FontWeight','bold');
xtickangle(ax,xtickRotation);

% ---- text label with white background ----------------------------------
hold on
offset = 0.05*range(ylim(ax));
for i = 1:nM
    text(i, medTime(i)+offset, sprintf('%.2f',medTime(i)), ...
        'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
        'FontSize',12, 'FontWeight','bold', ...
        'BackgroundColor','w', 'Margin',0.1);
end

hold off

% ---- figure sizing + export -------------------------------------------
fig2.Units = 'inches';
fig2.Position = [1 1 w hFig];
fig2.PaperUnits = 'inches';
fig2.PaperPosition = [0 0 w hFig];
fig2.PaperSize = [w hFig];
fig2.PaperPositionMode = 'manual';
exportgraphics(fig2,'Time_boxplot.pdf', ...
                   'ContentType','vector', ...
                   'BackgroundColor','none');

end
% -------------------------------------------------------------------------
% Helper: evaluate one held‑out fold  (estimator passed in)
% -------------------------------------------------------------------------
function [rmse, accuracy] = evaluate_fold(testData, model, estFcn)

    mse        = 0;   nPred = 0;
    correctLab = 0;   totLab = 0;

    for tr = 1:size(testData,1)
        localModel = model;          % fresh copy per trial
        for dir = 1:8
            decoded = [];
            times   = 320:20:size(testData(tr,dir).spikes,2);

            for t = times
                sample.trialId        = testData(tr,dir).trialId;
                sample.spikes         = testData(tr,dir).spikes(:,1:t);
                sample.decodedHandPos = decoded;
                sample.startHandPos   = testData(tr,dir).handPos(1:2,1);

                nOut = nargout(estFcn);
                if nOut==3
                    [xHat,yHat,newModel] = estFcn(sample, localModel);
                    localModel = newModel;
                else
                    [xHat,yHat] = estFcn(sample, localModel);
                end

                decoded = [decoded [xHat;yHat]];
                mse     = mse + norm(testData(tr,dir).handPos(1:2,t)-[xHat;yHat])^2;
            end
            nPred = nPred + numel(times);

            if isfield(localModel,'actualLabel') && localModel.actualLabel==dir
                correctLab = correctLab + 1;
            end
            totLab = totLab + 1;
        end
    end

    rmse     = sqrt(mse/nPred);
    accuracy = correctLab/totLab;
end
