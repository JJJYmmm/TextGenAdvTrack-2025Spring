"""
Utility functions for LLM detection models
"""

def adjust_prediction_score(pred_label, pred_score, model_type):
    """
    Adjusts the prediction score to a standardized format where higher values
    indicate higher likelihood of human-written text.
    
    Args:
        pred_label (str): Prediction label from the model
        pred_score (float): Prediction score/probability
        model_type (str): Model type ("your model")
        
    Returns:
        float: Adjusted score where higher values indicate more likely human-written text
    """
    if model_type == "argugpt":
        # ArguGPT: LABEL_1=machine(0), LABEL_0=human(1)
        if pred_label == "LABEL_0":  # Predicted as human
            return pred_score
        else:  # LABEL_1, predicted as machine
            return 1 - pred_score
        
    # if model_type == "your model":
    #     # your model: xxx=machine(0), yyy=human(1)
    #     if pred_label == "xxx":  # Predicted as human
    #         return pred_score
    #     else:  # yyy, predicted as machine
    #         return 1 - pred_score
        
    else:
        # Default handling, return original score
        
        return pred_score
