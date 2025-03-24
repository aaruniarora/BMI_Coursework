function sweep_LDA_PCA()
    % Initialize LDA and PCA parameter ranges
    lda_range = 4;   % Adjust this range
    pca_range = 35;  % Adjust this range
    alpha_range = 0.2;
    alpha2_range = 0.1;
    k_range = 5;
    pow_range = 2;
    
    % Ensure wandb is properly set up
    system('wandb login');  % Ensure logged in (only needed once)

    % Loop through all combinations of LDA and PCA
    for lda_dim = lda_range
        for pca_dim = pca_range
            for alpha = alpha_range
                for alpha2 = alpha2_range
                    for pow = pow_range
                        for k = k_range
                            fprintf('Running sweep: LDA = %d, PCA = %d, alpha = %d, alpha2 = %d, pow = %d, k = %d\n', lda_dim, pca_dim, alpha, alpha2, pow, k);
                
                            % Ensure model training is using the specified parameters
                            global LDADIM PCADIM ALPHA ALPHA2 POW K;
                            LDADIM = lda_dim;
                            PCADIM = pca_dim;
                            ALPHA = alpha;
                            ALPHA2 = alpha2;
                            POW = pow;
                            K = k;
            
                
                            % Load data and train model
                            load monkeydata_training.mat
                            rng(2013);
                            ix = randperm(length(trial));
                            trainingData = trial(ix(1:50), :);
                            testData = trial(ix(51:end), :);
                
                            modelParameters = positionEstimatorTraining(trainingData);
                            RMSE = testFunction_for_students_MTb(modelParameters);
                
                            % Log to Weights & Biases (wandb)
                            log_to_wandb(lda_dim, pca_dim, alpha, alpha2, pow, k, RMSE);
                        end
                    end
                end
            end
        end
    end
end

function log_to_wandb(lda_dim, pca_dim, alpha, alpha2, pow, k, RMSE)
    % Ensure wandb is initialized and logs the sweep parameters
    cmd = sprintf(['python -c "import wandb; ' ...
                   'wandb.init(project=''Aaruni''); ' ...
                   'wandb.log({''lda_dim'': %d, ''pca_dim'': %d, ''alpha'': %d, ''alpha2'': %d, ''pow'': %d, ''k'': %d,''rmse'': %.4f});"'], ...
                   lda_dim, pca_dim, alpha, alpha2, pow, k, RMSE);
    system(cmd);
end
