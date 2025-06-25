import os
import pickle

import pandas as pd
import rpy2.robjects as robjects

from estimation import run_estimation_methods
from utils import format_estimation_df


def run_py_estimation(dataset_paths, py_estimation_df_path):
    run_estimation = not os.path.isfile(py_estimation_df_path)
    method_names = ['NaiveLinReg', 'LinReg', 'RF', 'CausalForest', 'CausalForestDML', 'LinearDML', 'LinearDR', 'ForestDR', 'TNet', 'SNet1']

    y_name = 'logP(X13=0)'
    outcome_index = 13
    treatment_index = 12
    possible_treatment_values = [0, 1]
    possible_outcome_values = [0, 1, 2, 3]
    treatment_name = f'X{treatment_index}'
    all_covariates = [f'X{i}' for i in range(12)]
    observed_covariates = [f'X{i}' for i in range(4, 12)]
    
    if run_estimation:
        all_prediction_dfs = []
        all_estimation_dfs = []
        for covariate_names in [all_covariates, observed_covariates]:
            predictions_df = run_estimation_methods(
                dataset_paths=dataset_paths,
                method_names=method_names,
                covariate_names=covariate_names,
                treatment_name=treatment_name,
                y_name=y_name,
                outcome_index=outcome_index,
                possible_outcome_values=possible_outcome_values,
                treatment_index=treatment_index,
                possible_treatment_values=possible_treatment_values
            )
            predictions_df['model_name'] = predictions_df['df_path'].apply(lambda x: x.split('/')[2])
            all_prediction_dfs.append(predictions_df)
            all_estimation_dfs.append(format_estimation_df(predictions_df=predictions_df))
        predictions_df = pd.concat(all_prediction_dfs, axis=0)
        py_estimation_df = pd.concat(all_estimation_dfs, axis=0)
        with open(py_estimation_df_path, 'wb') as file:
            pickle.dump(py_estimation_df, file)
        print(f'Saved at: {py_estimation_df_path}')
    else:
        with open(py_estimation_df_path, 'rb') as file:
            py_estimation_df = pickle.load(file)
        print(f'Loaded from: {py_estimation_df_path}')
    
    return py_estimation_df


def run_py_estimation_1k():
    py_estimation_df_path = 'py_estimation_df.pkl'
    dataset_paths = pd.read_csv('all_df_paths.csv')['df_path'].values.tolist()
    return run_py_estimation(dataset_paths=dataset_paths, py_estimation_df_path=py_estimation_df_path)


def run_py_estimation_10k():
    py_estimation_df_path = 'py_10k_estimation_df.pkl'
    dataset_paths = pd.read_csv('10k_df_paths.csv')['df_path'].values.tolist()
    return run_py_estimation(dataset_paths=dataset_paths, py_estimation_df_path=py_estimation_df_path)


def run_r_estimation_1k():
    r_estimation_results_all_path = 'r_predictions_df_all.pkl'
    r_estimation_results_hidden_path = 'r_predictions_df_hidden.pkl'

    run_r_estimation_all = not os.path.isfile(r_estimation_results_hidden_path)
    run_r_estimation_hidden = not os.path.isfile(r_estimation_results_hidden_path)

    if run_r_estimation_all:
        robjects.r(
            """
                bartcause_ITE_CI <- function(X, Y, T, Xtest, alpha=0.05){
                bartfit <- bartCause::bartc(response=Ytrain, treatment=Ttrain, confounders=Xtrain, keepTrees=FALSE)
                sates <- bartCause::extract(bartfit, type='sate')
                cates <- bartCause::extract(bartfit, type='icate')
                ites <- bartCause::extract(bartfit, type='ite')
                
                sate <- mean(sates)
                
                qnorm_value <- qnorm(1 - alpha/2)
                cate.mean <- apply(cates, 2, mean)
                cate.sd <- apply(cates, 2, sd)
                cate.lb <- cate.mean - qnorm_value * cate.sd
                cate.ub <- cate.mean + qnorm_value * cate.sd
                
                ite.mean <- apply(ites, 2, mean)
                ite.sd <- apply(ites, 2, sd)
                ite.lb <- ite.mean - qnorm_value * ite.sd
                ite.ub <- ite.mean + qnorm_value * ite.sd
                
                return(as.data.frame(list(
                    ate_BART=sate, cate_BART=cate.mean, cate_lower_bounds_BART=cate.lb, cate_upper_bounds_BART=cate.ub, 
                    ite_BART=ite.mean, ite_lower_bounds_BART=ite.lb, ite_upper_bounds_BART=ite.ub
                )))
                }

                set.seed(1)
                all_df_paths <- readr::read_csv('all_df_paths.csv')['df_path'] |> unlist()
                r_predictions <- list()
                bart_col_names <- c(
                    'ite_BART', 'ite_lower_bounds_BART', 'ite_upper_bounds_BART', 
                    'cate_BART', 'cate_lower_bounds_BART', 'cate_upper_bounds_BART',
                    'ate_BART'
                )
                col_names <- c(
                    'ite_lower_bounds_Conformal', 'ite_upper_bounds_Conformal', 
                    bart_col_names
                )
                for (path in all_df_paths){
                    df <- data.frame(matrix(NA, nrow=1000, ncol=length(col_names)))
                    names(df) <- col_names
                    r_predictions[[path]] <- df
                }
                i <- 0
                for (df_path in all_df_paths){
                    if (i %% 10 == 0){
                    print(paste0(i, '/', length(all_df_paths)))
                    }
                    data_df <- read.csv(df_path)
                    covariate_names <- paste0('X', 0:11)
                    treatment_name <- 'X12'
                    y_name <- 'logP.X13.0.'
                    
                    Xtrain <- data_df[covariate_names]
                    Ttrain <- data_df[treatment_name] |> unlist()
                    Ytrain <- data_df[y_name] |> unlist()
                    
                    alpha <- 0.05
                    quantiles <- c(alpha/2, 1 - alpha/2)
                    ci_function <- cfcausal::conformalIte(X=Xtrain, Y=Ytrain, T=Ttrain, algo='counterfactual', alpha=alpha, quantiles=quantiles)
                    conformal_ite_intervals <- ci_function(X=Xtrain, Y=Ytrain, T=Ttrain)
                    r_predictions[[df_path]][, 'ite_lower_bounds_Conformal'] <- conformal_ite_intervals['lower']
                    r_predictions[[df_path]][, 'ite_upper_bounds_Conformal'] <- conformal_ite_intervals['upper']
                    
                    bart_intervals <- bartcause_ITE_CI(X=Xtrain, Y=Ytrain, T=Ttrain, alpha=alpha)
                    r_predictions[[df_path]][bart_col_names] <- bart_intervals[,bart_col_names]
                    i <- i + 1
                }
                print('Done.')
            """
        )
        r_predictions_all = robjects.globalenv['r_predictions']
        r_predictions_df_all = pd.DataFrame(r_predictions_all, index=r_predictions_all.names, columns=r_predictions_all[0].names).reset_index(names='df_path')
        r_predictions_df_all['covariate_names'] = [tuple([f'X{i}' for i in range(12)])] * r_predictions_df_all.shape[0]
        r_predictions_df_all['treatment_name'] = 'X12'
        r_predictions_df_all['outcome_name'] = 'logP(X13=0)'
        with open(r_estimation_results_all_path, 'wb') as file:
            pickle.dump(r_predictions_df_all, file)
        print(f'Saved at: {r_estimation_results_all_path}')
    else:
        with open(r_estimation_results_all_path, 'rb') as file:
            r_predictions_df_all = pickle.load(file)
            print(f'Loaded from: {r_estimation_results_all_path}')
    
    if run_r_estimation_hidden:
        robjects.r(
            """
                bartcause_ITE_CI <- function(X, Y, T, Xtest, alpha=0.05){
                bartfit <- bartCause::bartc(response=Ytrain, treatment=Ttrain, confounders=Xtrain, keepTrees=FALSE)
                sates <- bartCause::extract(bartfit, type='sate')
                cates <- bartCause::extract(bartfit, type='icate')
                ites <- bartCause::extract(bartfit, type='ite')
                
                sate <- mean(sates)
                
                qnorm_value <- qnorm(1 - alpha/2)
                cate.mean <- apply(cates, 2, mean)
                cate.sd <- apply(cates, 2, sd)
                cate.lb <- cate.mean - qnorm_value * cate.sd
                cate.ub <- cate.mean + qnorm_value * cate.sd
                
                ite.mean <- apply(ites, 2, mean)
                ite.sd <- apply(ites, 2, sd)
                ite.lb <- ite.mean - qnorm_value * ite.sd
                ite.ub <- ite.mean + qnorm_value * ite.sd
                
                return(as.data.frame(list(
                    ate_BART=sate, cate_BART=cate.mean, cate_lower_bounds_BART=cate.lb, cate_upper_bounds_BART=cate.ub, 
                    ite_BART=ite.mean, ite_lower_bounds_BART=ite.lb, ite_upper_bounds_BART=ite.ub
                )))
                }

                set.seed(1)
                all_df_paths <- readr::read_csv('all_df_paths.csv')['df_path'] |> unlist()
                r_predictions <- list()
                bart_col_names <- c(
                    'ite_BART', 'ite_lower_bounds_BART', 'ite_upper_bounds_BART', 
                    'cate_BART', 'cate_lower_bounds_BART', 'cate_upper_bounds_BART',
                    'ate_BART'
                )
                col_names <- c(
                    'ite_lower_bounds_Conformal', 'ite_upper_bounds_Conformal', 
                    bart_col_names
                )
                for (path in all_df_paths){
                    df <- data.frame(matrix(NA, nrow=1000, ncol=length(col_names)))
                    names(df) <- col_names
                    r_predictions[[path]] <- df
                }
                i <- 0
                for (df_path in all_df_paths){
                    if (i %% 10 == 0){
                    print(paste0(i, '/', length(all_df_paths)))
                    }
                    data_df <- read.csv(df_path)
                    covariate_names <- paste0('X', 4:11)
                    treatment_name <- 'X12'
                    y_name <- 'logP.X13.0.'
                    
                    Xtrain <- data_df[covariate_names]
                    Ttrain <- data_df[treatment_name] |> unlist()
                    Ytrain <- data_df[y_name] |> unlist()
                    
                    alpha <- 0.05
                    quantiles <- c(alpha/2, 1 - alpha/2)
                    ci_function <- cfcausal::conformalIte(X=Xtrain, Y=Ytrain, T=Ttrain, algo='counterfactual', alpha=alpha, quantiles=quantiles)
                    conformal_ite_intervals <- ci_function(X=Xtrain, Y=Ytrain, T=Ttrain)
                    r_predictions[[df_path]][, 'ite_lower_bounds_Conformal'] <- conformal_ite_intervals['lower']
                    r_predictions[[df_path]][, 'ite_upper_bounds_Conformal'] <- conformal_ite_intervals['upper']
                    
                    bart_intervals <- bartcause_ITE_CI(X=Xtrain, Y=Ytrain, T=Ttrain, alpha=alpha)
                    r_predictions[[df_path]][bart_col_names] <- bart_intervals[,bart_col_names]
                    i <- i + 1
                }
                print('Done.')
            """
        )
        r_predictions_hidden = robjects.globalenv['r_predictions']
        r_predictions_df_hidden = pd.DataFrame(r_predictions_hidden, index=r_predictions_hidden.names, columns=r_predictions_hidden[0].names).reset_index(names='df_path')
        r_predictions_df_hidden['covariate_names'] = [tuple([f'X{i}' for i in range(4, 12)])] * r_predictions_df_hidden.shape[0]
        r_predictions_df_hidden['treatment_name'] = 'X12'
        r_predictions_df_hidden['outcome_name'] = 'logP(X13=0)'
        with open(r_estimation_results_hidden_path, 'wb') as file:
            pickle.dump(r_predictions_df_hidden, file)
        print(f'Saved at: {r_estimation_results_hidden_path}')
    else:
        with open(r_estimation_results_hidden_path, 'rb') as file:
            r_predictions_df_hidden = pickle.load(file)
        print(f'Loaded from: {r_estimation_results_hidden_path}')
    
    return r_predictions_df_all, r_predictions_df_hidden


def run_r_estimation_10k():
    r_estimation_results_all_path = 'r_10k_predictions_df_all.pkl'
    r_estimation_results_hidden_path = 'r_10k_predictions_df_hidden.pkl'

    run_r_estimation_all = not os.path.isfile(r_estimation_results_hidden_path)
    run_r_estimation_hidden = not os.path.isfile(r_estimation_results_hidden_path)

    if run_r_estimation_all:
        robjects.r(
            """
                bartcause_ITE_CI <- function(X, Y, T, Xtest, alpha=0.05){
                bartfit <- bartCause::bartc(response=Ytrain, treatment=Ttrain, confounders=Xtrain, keepTrees=FALSE)
                sates <- bartCause::extract(bartfit, type='sate')
                cates <- bartCause::extract(bartfit, type='icate')
                ites <- bartCause::extract(bartfit, type='ite')
                
                sate <- mean(sates)
                
                qnorm_value <- qnorm(1 - alpha/2)
                cate.mean <- apply(cates, 2, mean)
                cate.sd <- apply(cates, 2, sd)
                cate.lb <- cate.mean - qnorm_value * cate.sd
                cate.ub <- cate.mean + qnorm_value * cate.sd
                
                ite.mean <- apply(ites, 2, mean)
                ite.sd <- apply(ites, 2, sd)
                ite.lb <- ite.mean - qnorm_value * ite.sd
                ite.ub <- ite.mean + qnorm_value * ite.sd
                
                return(as.data.frame(list(
                    ate_BART=sate, cate_BART=cate.mean, cate_lower_bounds_BART=cate.lb, cate_upper_bounds_BART=cate.ub, 
                    ite_BART=ite.mean, ite_lower_bounds_BART=ite.lb, ite_upper_bounds_BART=ite.ub
                )))
                }

                set.seed(1)
                all_df_paths <- readr::read_csv('10k_df_paths.csv')['df_path'] |> unlist()
                r_predictions <- list()
                bart_col_names <- c(
                    'ite_BART', 'ite_lower_bounds_BART', 'ite_upper_bounds_BART', 
                    'cate_BART', 'cate_lower_bounds_BART', 'cate_upper_bounds_BART',
                    'ate_BART'
                )
                col_names <- c(
                    'ite_lower_bounds_Conformal', 'ite_upper_bounds_Conformal', 
                    bart_col_names
                )
                for (path in all_df_paths){
                    df <- data.frame(matrix(NA, nrow=10000, ncol=length(col_names)))
                    names(df) <- col_names
                    r_predictions[[path]] <- df
                }
                i <- 0
                for (df_path in all_df_paths){
                    if (i %% 10 == 0){
                    print(paste0(i, '/', length(all_df_paths)))
                    }
                    data_df <- read.csv(df_path)
                    covariate_names <- paste0('X', 0:11)
                    treatment_name <- 'X12'
                    y_name <- 'logP.X13.0.'
                    
                    Xtrain <- data_df[covariate_names]
                    Ttrain <- data_df[treatment_name] |> unlist()
                    Ytrain <- data_df[y_name] |> unlist()
                    
                    alpha <- 0.05
                    quantiles <- c(alpha/2, 1 - alpha/2)
                    ci_function <- cfcausal::conformalIte(X=Xtrain, Y=Ytrain, T=Ttrain, algo='counterfactual', alpha=alpha, quantiles=quantiles)
                    conformal_ite_intervals <- ci_function(X=Xtrain, Y=Ytrain, T=Ttrain)
                    r_predictions[[df_path]][, 'ite_lower_bounds_Conformal'] <- conformal_ite_intervals['lower']
                    r_predictions[[df_path]][, 'ite_upper_bounds_Conformal'] <- conformal_ite_intervals['upper']
                    
                    bart_intervals <- bartcause_ITE_CI(X=Xtrain, Y=Ytrain, T=Ttrain, alpha=alpha)
                    r_predictions[[df_path]][bart_col_names] <- bart_intervals[,bart_col_names]
                    i <- i + 1
                }
                print('Done.')
            """
        )
        r_predictions_all = robjects.globalenv['r_predictions']
        r_predictions_df_all = pd.DataFrame(r_predictions_all, index=r_predictions_all.names, columns=r_predictions_all[0].names).reset_index(names='df_path')
        r_predictions_df_all['covariate_names'] = [tuple([f'X{i}' for i in range(12)])] * r_predictions_df_all.shape[0]
        r_predictions_df_all['treatment_name'] = 'X12'
        r_predictions_df_all['outcome_name'] = 'logP(X13=0)'
        with open(r_estimation_results_all_path, 'wb') as file:
            pickle.dump(r_predictions_df_all, file)
        print(f'Saved at: {r_estimation_results_all_path}')
    else:
        with open(r_estimation_results_all_path, 'rb') as file:
            r_predictions_df_all = pickle.load(file)
            print(f'Loaded from: {r_estimation_results_all_path}')
    
    if run_r_estimation_hidden:
        robjects.r(
            """
                bartcause_ITE_CI <- function(X, Y, T, Xtest, alpha=0.05){
                bartfit <- bartCause::bartc(response=Ytrain, treatment=Ttrain, confounders=Xtrain, keepTrees=FALSE)
                sates <- bartCause::extract(bartfit, type='sate')
                cates <- bartCause::extract(bartfit, type='icate')
                ites <- bartCause::extract(bartfit, type='ite')
                
                sate <- mean(sates)
                
                qnorm_value <- qnorm(1 - alpha/2)
                cate.mean <- apply(cates, 2, mean)
                cate.sd <- apply(cates, 2, sd)
                cate.lb <- cate.mean - qnorm_value * cate.sd
                cate.ub <- cate.mean + qnorm_value * cate.sd
                
                ite.mean <- apply(ites, 2, mean)
                ite.sd <- apply(ites, 2, sd)
                ite.lb <- ite.mean - qnorm_value * ite.sd
                ite.ub <- ite.mean + qnorm_value * ite.sd
                
                return(as.data.frame(list(
                    ate_BART=sate, cate_BART=cate.mean, cate_lower_bounds_BART=cate.lb, cate_upper_bounds_BART=cate.ub, 
                    ite_BART=ite.mean, ite_lower_bounds_BART=ite.lb, ite_upper_bounds_BART=ite.ub
                )))
                }

                set.seed(1)
                all_df_paths <- readr::read_csv('10k_df_paths.csv')['df_path'] |> unlist()
                r_predictions <- list()
                bart_col_names <- c(
                    'ite_BART', 'ite_lower_bounds_BART', 'ite_upper_bounds_BART', 
                    'cate_BART', 'cate_lower_bounds_BART', 'cate_upper_bounds_BART',
                    'ate_BART'
                )
                col_names <- c(
                    'ite_lower_bounds_Conformal', 'ite_upper_bounds_Conformal', 
                    bart_col_names
                )
                for (path in all_df_paths){
                    df <- data.frame(matrix(NA, nrow=10000, ncol=length(col_names)))
                    names(df) <- col_names
                    r_predictions[[path]] <- df
                }
                i <- 0
                for (df_path in all_df_paths){
                    if (i %% 10 == 0){
                    print(paste0(i, '/', length(all_df_paths)))
                    }
                    data_df <- read.csv(df_path)
                    covariate_names <- paste0('X', 4:11)
                    treatment_name <- 'X12'
                    y_name <- 'logP.X13.0.'
                    
                    Xtrain <- data_df[covariate_names]
                    Ttrain <- data_df[treatment_name] |> unlist()
                    Ytrain <- data_df[y_name] |> unlist()
                    
                    alpha <- 0.05
                    quantiles <- c(alpha/2, 1 - alpha/2)
                    ci_function <- cfcausal::conformalIte(X=Xtrain, Y=Ytrain, T=Ttrain, algo='counterfactual', alpha=alpha, quantiles=quantiles)
                    conformal_ite_intervals <- ci_function(X=Xtrain, Y=Ytrain, T=Ttrain)
                    r_predictions[[df_path]][, 'ite_lower_bounds_Conformal'] <- conformal_ite_intervals['lower']
                    r_predictions[[df_path]][, 'ite_upper_bounds_Conformal'] <- conformal_ite_intervals['upper']
                    
                    bart_intervals <- bartcause_ITE_CI(X=Xtrain, Y=Ytrain, T=Ttrain, alpha=alpha)
                    r_predictions[[df_path]][bart_col_names] <- bart_intervals[,bart_col_names]
                    i <- i + 1
                }
                print('Done.')
            """
        )
        r_predictions_hidden = robjects.globalenv['r_predictions']
        r_predictions_df_hidden = pd.DataFrame(r_predictions_hidden, index=r_predictions_hidden.names, columns=r_predictions_hidden[0].names).reset_index(names='df_path')
        r_predictions_df_hidden['covariate_names'] = [tuple([f'X{i}' for i in range(4, 12)])] * r_predictions_df_hidden.shape[0]
        r_predictions_df_hidden['treatment_name'] = 'X12'
        r_predictions_df_hidden['outcome_name'] = 'logP(X13=0)'
        with open(r_estimation_results_hidden_path, 'wb') as file:
            pickle.dump(r_predictions_df_hidden, file)
        print(f'Saved at: {r_estimation_results_hidden_path}')
    else:
        with open(r_estimation_results_hidden_path, 'rb') as file:
            r_predictions_df_hidden = pickle.load(file)
        print(f'Loaded from: {r_estimation_results_hidden_path}')
    
    return r_predictions_df_all, r_predictions_df_hidden
