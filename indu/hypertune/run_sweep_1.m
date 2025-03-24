function sweep_k()
    % Initialize parameter ranges
    % alpha_range = 0.1:0.1:0.5;
    alpha_range = 0.35;
    lda_range = 4:2:8;
    pca_thres_range = 0.3:0.1:0.9;
    % alpha_est_range = 0.1:0.1:0.5;
    alpha_est_range = 0.3;
    % pow_range = 1:1:3;
    pow_range = 1;
    k_range = 10:5:30;
    len_b_mode_range = 2:2:12;

    % Ensure wandb is properly set up
    system('wandb login');  % Ensure logged in (only needed once)

    % Loop through all combinations of parameters
for alpha = alpha_range
    for lda = lda_range
        for pca_thres = pca_thres_range
            for alpha_est = alpha_est_range
                for pow = pow_range
                    for k = k_range
                        for len_b = len_b_mode_range

                                    fprintf('Running sweep: pca_thres = %.3f',pca_thres);
                        
                                    % Ensure model training uses the specified parameters
                                    global ALPHA LDA_DIM PCA_T ALPHA_E K POW LEN_B_MODE
                                    ALPHA = alpha;
                                    LDA_DIM = lda;
                                    PCA_T = pca_thres;
                                    ALPHA_E = alpha_est;
                                    K = k;
                                    POW = pow;
                                    LEN_B_MODE = len_b;

                        
                                    % % Load data and train model
                                    % load monkeydata_training.mat
                            
                                    % ix = randperm(length(trial));
                                    % trainingData = trial(ix(1:50), :);
                                    % testData = trial(ix(51:end), :);
                        
                                    % modelParameters = positionEstimatorTraining(trainingData);
                                    [RMSE,mean_acc,elapsedTime] = testFunction_for_students_MTb_k(10);
                        
                                    % Log to Weights & Biases (wandb)
                                    log_to_wandb(alpha,lda,pca_thres,alpha_est,k,pow,len_b, RMSE,mean_acc,elapsedTime);
                                end
                            end
                end
            end
        end
    end
end
end

function   log_to_wandb(alpha,lda,pca_thres,alpha_est,k,pow,len_b, RMSE,mean_acc,elapsedTime)
    % Ensure wandb is initialized and logs the sweep parameters
    % cmd = sprintf(['python -c "import wandb; ' ...
      cmd = sprintf(['"C:\\Users\\Indum\\AppData\\Local\\Programs\\Python\\Python311\\python.exe" -c "import wandb;'...
                   'wandb.init(project=''hi''); ' ...
                   'wandb.log({''alpha'': %.2f, ''lda'': %.2f, ''pca_thres'': %.2f,''alpha_est'':%.2f,''k'':%.2f,''pow'':%.2f,''len_b'':%.2f,''RMSE'':%.4f,''mean_acc'':%.2f,''elapsedTime'':%.2f});"'], ...
                     alpha,lda,pca_thres,alpha_est,k,pow,len_b, RMSE,mean_acc,elapsedTime);
    system(cmd);
end
