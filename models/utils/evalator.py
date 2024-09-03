import numpy as np
import torch
from sklearn.metrics import roc_auc_score,r2_score,accuracy_score,mean_squared_error,matthews_corrcoef,confusion_matrix
class CHEMBLEvaluator:
    def __init__(self):
        '''
            Evaluator for the PCQM4M dataset
            Metric is Mean Absolute Error
        '''
        pass 

    def eval(self, input_dict):
        '''
            y_true: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_pred: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_true and y_pred need to be of the same type (either numpy.ndarray or torch.Tensor)
        '''

        y_pred_m, y_true_m,y_pred_r,y_true_r,y_pred_c,y_true_c \
            =   input_dict['y_pred_m'], input_dict['y_true_m'],input_dict['y_pred_r'], \
            input_dict['y_true_r'],input_dict['y_pred_c'], input_dict['y_true_c']
        
        # y_pred_m=torch.argmax(y_pred_m,dim=2)
        y_pred_c=torch.where(y_pred_c <0.5,0,1)

        return {'auc': accuracy_score(y_true_m.view(-1).cpu() , y_pred_m.view(-1).cpu()).item()+roc_auc_score(y_true_c.cpu() , y_pred_c.cpu()).item()+r2_score(y_true_r.cpu() ,y_pred_r.cpu()).item()
        }


class CLFEvaluator:
    def __init__(self):
        '''
            Evaluator for the PCQM4M dataset
            Metric is Mean Absolute Error
        '''
        pass 

    def eval(self, input_dict):
        '''
            y_true: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_pred: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_true and y_pred need to be of the same type (either numpy.ndarray or torch.Tensor)
        '''


        y_pred, y_true = input_dict['y_pred_c'], input_dict['y_true_c']
        y_pred=torch.sigmoid(y_pred)
        #print(y_true)
        y_prob=y_pred
        y_pred=torch.where(y_pred <0.50,0,1)
        if isinstance(y_true, torch.Tensor):
            # print(y_pred.shape,y_true.shape)
            cm=confusion_matrix(y_true.cpu().squeeze() , y_pred.cpu().squeeze())
            TP=cm[1,1]
            TN=cm[0,0]
            FP=cm[0,1]
            FN=cm[1,0]
            return {'auc': roc_auc_score(y_true.cpu() , y_prob.cpu()).item()
                    ,'acc': accuracy_score(y_true.cpu() , y_pred.cpu()).item()
                    ,'mcc':  matthews_corrcoef(y_true.cpu() , y_pred.cpu())
                    ,'specificty': TN/(TN+FP)
                    ,'sensitivity': TP/(TP+FN)

            }
        else:
            return {'auc': float(roc_auc_score(y_true , y_pred))}


class REGEvaluator:
    def __init__(self):
        '''
            Evaluator for the PCQM4M dataset
            Metric is Mean Absolute Error
        '''
        pass 

    def eval(self, input_dict):
        '''
            y_true: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_pred: numpy.ndarray or torch.Tensor of shape (num_graphs,)
            y_true and y_pred need to be of the same type (either numpy.ndarray or torch.Tensor)
        '''
        y_pred, y_true = input_dict['y_pred_r'], input_dict['y_true_r']
#torch.sqrt(torch.mean(torch.square(y_pred - y_true))).cpu().item()
        if isinstance(y_true, torch.Tensor):
            return {'mae': torch.mean(torch.abs(y_pred - y_true)).cpu().item()
        ,'rmse':mean_squared_error(y_true.cpu(),y_pred.cpu()),'r2': r2_score(y_true.cpu(),y_pred.cpu())}
        else:
            return {'mae': float(np.mean(np.absolute(y_pred - y_true)))
        ,'rmse':np.sqrt(mean_squared_error(y_true,y_pred)),'r2': r2_score(y_true,y_pred)}
 
