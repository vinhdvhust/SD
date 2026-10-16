===========================================================================
  INCEPTION-RESNET-V2 SMOKER CLASSIFIER - ARCHITECTURE SUMMARY
===========================================================================
[Backbone]             : Inception-ResNet-V2 (164 layers deep, pretrained)
[Device]               : CUDA:0
[Input Resolution]     : (3, 224, 224) (Batch Size = 2)
[Feature Map Out]      : (1536, 5, 5) -> Global Avg Pool -> (1536,)
---------------------------------------------------------------------------
Custom Classification Head (Section 4.1 & Figure 3):
  1. Linear(1536 -> 1536)  + ReLU + Dropout(p=0.5)
  2. Linear(1536 -> 1536)  + ReLU + Dropout(p=0.5)
  3. Linear(1536 -> 512)   + ReLU + Dropout(p=0.5)
  4. Linear(512  -> 1)     + Sigmoid (Binary Output: NotSmoking=0, Smoking=1)
---------------------------------------------------------------------------
[Total Parameters]     : 59,815,585
[Frozen Parameters]    : 54,306,464 (Conv base feature extractor)
[Trainable Parameters] : 5,509,121 (Custom FC dense head)
---------------------------------------------------------------------------
[Forward Test Output]  : Logits shape = (2, 1)
[Sigmoid Probs Output] : Range = [1.0000, 1.0000]
===========================================================================


===========================================================================
  SMOKER DETECTION SYSTEM - TRAINING PIPELINE
===========================================================================
[Device]           : CUDA
[Epochs]           : 50
[Batch Size]       : 64
[Learning Rate]    : 0.01
[Optimizer]        : SGD (momentum=0.9, weight_decay=0.0001)
[Model Save Dir]   : /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/model
[Outputs Dir]      : /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/outputs
===========================================================================

[1/4] Loading Datasets...
  Training samples   : 716 (12 batches)
  Validation samples : 180 (3 batches)
  Testing samples    : 224 (4 batches)

[2/4] Initializing Pretrained Inception-ResNet-V2...
  Backbone           : Inception-ResNet-V2 (Frozen: 54,306,464 params)
  Trainable Head     : 3-Layer Dense Head (5,509,121 params)

[3/4] Starting Training Execution...
-------------------------------------------------------------------------------------
 Epoch  | Train Loss  |  Train Acc  |  Val Loss   |   Val Acc   |    LR     |  Time   | Status
-------------------------------------------------------------------------------------
[01/50] |   40.9522   |   50.00   % |   5.4429    |   50.00   % |  0.01000  |  4.6  s | BEST                       
[02/50] |   4.3430    |   50.56   % |   0.6894    |   53.89   % |  0.01000  |  5.1  s | BEST                       
[03/50] |   1.0922    |   48.60   % |   0.6854    |   53.33   % |  0.01000  |  4.0  s |                            
[04/50] |   0.7360    |   49.30   % |   0.6867    |   53.33   % |  0.01000  |  4.0  s |                            
[05/50] |   0.7208    |   50.56   % |   0.6904    |   50.00   % |  0.01000  |  4.0  s |                            
[06/50] |   0.7018    |   50.56   % |   0.6877    |   51.67   % |  0.01000  |  4.9  s |                            
[07/50] |   0.6973    |   50.56   % |   0.6843    |   53.33   % |  0.01000  |  4.0  s |                            
[08/50] |   0.6922    |   51.82   % |   0.6785    |   76.67   % |  0.01000  |  4.0  s | BEST                       
[09/50] |   0.6958    |   51.40   % |   0.6805    |   75.00   % |  0.01000  |  4.0  s |                            
[10/50] |   0.6881    |   52.79   % |   0.6862    |   50.00   % |  0.01000  |  6.3  s |                            
[11/50] |   0.6922    |   52.93   % |   0.6832    |   55.56   % |  0.01000  |  4.0  s |                            
[12/50] |   0.6880    |   56.28   % |   0.6840    |   52.22   % |  0.01000  |  4.0  s |                            
[13/50] |   0.6771    |   56.70   % |   0.6847    |   50.00   % |  0.01000  |  4.1  s |                            
[14/50] |   0.6776    |   58.94   % |   0.6580    |   78.33   % |  0.01000  |  4.0  s | BEST                       
[15/50] |   0.6769    |   59.50   % |   0.6360    |   78.89   % |  0.01000  |  4.1  s | BEST                       
[16/50] |   0.6617    |   60.47   % |   0.6340    |   71.67   % |  0.01000  |  4.2  s |                            
[17/50] |   0.6561    |   59.92   % |   0.6036    |   80.56   % |  0.01000  |  4.1  s | BEST                       
[18/50] |   0.6617    |   62.29   % |   0.5769    |   84.44   % |  0.01000  |  4.1  s | BEST                       
[19/50] |   0.6315    |   66.48   % |   0.5480    |   83.33   % |  0.01000  |  4.1  s |                            
[20/50] |   0.6410    |   62.71   % |   0.5239    |   85.00   % |  0.01000  |  6.4  s | BEST                       
[21/50] |   0.6268    |   66.76   % |   0.5030    |   84.44   % |  0.01000  |  4.1  s |                            
[22/50] |   0.6068    |   68.30   % |   0.4835    |   85.00   % |  0.01000  |  4.1  s | BEST                       
[23/50] |   0.5931    |   70.11   % |   0.4727    |   83.89   % |  0.01000  |  4.1  s |                            
[24/50] |   0.6044    |   67.04   % |   0.4187    |   86.11   % |  0.01000  |  4.1  s | BEST                       
[25/50] |   0.5964    |   66.90   % |   0.4337    |   86.11   % |  0.01000  |  4.0  s |                            
[26/50] |   0.5548    |   71.37   % |   0.4026    |   85.00   % |  0.01000  |  4.9  s |                            
[27/50] |   0.5525    |   72.07   % |   0.3600    |   86.11   % |  0.01000  |  4.1  s | BEST                       
[28/50] |   0.5696    |   72.91   % |   0.3927    |   87.78   % |  0.01000  |  4.1  s | BEST                       
[29/50] |   0.5463    |   72.35   % |   0.3660    |   87.22   % |  0.01000  |  6.2  s |                            
[30/50] |   0.5427    |   75.00   % |   0.3526    |   87.22   % |  0.01000  |  5.0  s |                            
[31/50] |   0.5178    |   76.82   % |   0.3362    |   85.00   % |  0.01000  |  4.0  s |                            
[32/50] |   0.5220    |   73.04   % |   0.3467    |   87.22   % |  0.01000  |  4.1  s |                            
[33/50] |   0.5124    |   76.12   % |   0.3262    |   88.33   % |  0.01000  |  4.1  s | BEST                       
[34/50] |   0.4994    |   75.84   % |   0.3365    |   88.33   % |  0.01000  |  4.1  s |                            
[35/50] |   0.4748    |   76.96   % |   0.3108    |   87.78   % |  0.01000  |  6.2  s |                            
[36/50] |   0.5586    |   72.07   % |   0.3375    |   87.78   % |  0.01000  |  4.1  s |                            
[37/50] |   0.5283    |   74.30   % |   0.3460    |   88.89   % |  0.01000  |  4.1  s | BEST                       
[38/50] |   0.4999    |   76.12   % |   0.3104    |   90.56   % |  0.01000  |  4.2  s | BEST                       
[39/50] |   0.5150    |   75.00   % |   0.4077    |   83.89   % |  0.01000  |  4.1  s |                            
[40/50] |   0.5229    |   73.60   % |   0.3444    |   87.78   % |  0.01000  |  4.2  s |                            
[41/50] |   0.5032    |   77.23   % |   0.3281    |   87.22   % |  0.01000  |  4.2  s |                            
[42/50] |   0.5129    |   75.28   % |   0.3204    |   88.33   % |  0.01000  |  4.2  s |                            
[43/50] |   0.4761    |   77.79   % |   0.3702    |   85.00   % |  0.01000  |  4.2  s |                            
[44/50] |   0.4822    |   76.68   % |   0.3036    |   90.00   % |  0.01000  |  4.1  s |                            
[45/50] |   0.4933    |   76.26   % |   0.2953    |   87.78   % |  0.01000  |  4.1  s |                            
[46/50] |   0.4996    |   76.96   % |   0.2923    |   89.44   % |  0.01000  |  4.2  s |                            
[47/50] |   0.4650    |   77.09   % |   0.3408    |   88.89   % |  0.01000  |  4.1  s |                            
[48/50] |   0.4863    |   77.65   % |   0.3249    |   90.00   % |  0.01000  |  4.1  s |                            
[49/50] |   0.4715    |   77.09   % |   0.2852    |   90.00   % |  0.01000  |  4.2  s |                            
[50/50] |   0.4579    |   78.77   % |   0.2874    |   89.44   % |  0.01000  |  4.2  s |                            
-------------------------------------------------------------------------------------

[4/4] Training Complete in 4.91 minutes.
  Best Validation Accuracy : 90.56% (Loss: 0.3104)
  Best Model Checkpoint    : /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/model/best_model.pth
  Latest Checkpoint        : /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/model/latest_checkpoint.pth
  Training History JSON    : /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/outputs/training_history.json
[Curves] Learning curves saved to: /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/outputs/training_curves.png



===========================================================================
  SMOKER DETECTION SYSTEM - EVALUATION PIPELINE
===========================================================================
[Device]           : CUDA
[Checkpoint]       : best_model_50epochs.pth
[Output Directory] : /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/eval/50epochs
[Threshold]        : 0.5
===========================================================================

[1/3] Loading Testing Dataset (224 images)...
  Test samples loaded: 224 (4 batches)

[2/3] Initializing Model & Loading Weights...

[3/3] Running Inference & Calculating 11 Metrics...
Evaluating Test Set: 100%|████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.84batch/s]

===========================================================================
  EVALUATION RESULTS - 11 QUANTITATIVE METRICS (SECTION 6.2)
===========================================================================
  Accuracy (Acc)              : 89.73% (201/224 correct)
  Precision (PVp)             : 86.78%
  Recall / Sensitivity        : 93.75%
  Specificity                 : 85.71%
  Negative Pred Value (PVn)   : 93.20%
  False Positive Rate (FPR)   : 14.29% (Fp = 16)
  False Negative Rate (FNR)   : 6.25% (Fn = 7)
  False Discovery Rate (FDR)  : 13.22%
  F1-Score                    : 0.9013
  Error Rate (Er)             : 10.27%
  ROC Area Under Curve (AUC)  : 0.9621
  Average Precision (AP)      : 0.9632
===========================================================================

[Chart] Confusion Matrix saved to: /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/eval/50epochs/confusion_matrix.png
[Chart] ROC Curve saved to: /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/eval/50epochs/roc_curve.png
[Chart] Precision-Recall Curve saved to: /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/eval/50epochs/precision_recall_curve.png
[Report] JSON metrics archive saved to: /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/eval/50epochs/evaluation_metrics.json
[Report] Markdown comparison report saved to: /home/vinh/Hoc_tap/ThiGiacMayTinh/SD/eval/50epochs/metrics_comparison.md