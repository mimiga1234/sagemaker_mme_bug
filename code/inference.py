"""Serving Classification Model"""

import builtins
import shared.serving as serve

    
def model_fn(model_dir: str) -> dict:
    """Load model from previously saved artifact."""
    return serve.model_loading(model_dir, 'classification')


def transform_fn(
    modelDct: dict,
    request_body: str,
    input_content_type: str,
    output_content_type: str = "application/json",
) -> builtins.tuple[str, str]:
    """
    Transform function to process input data and generate predictions.

    Args:
        modelDct: The trained model object.
        request_body: The input data.
        input_content_type: The content type of the input data.
        output_content_type: The desired content type of the output.

    Returns:
        The prediction output and the corresponding content type.
    """
    output, output_content_type = serve.model_inference(
        modelDct=modelDct,
        request_body=request_body,
        input_content_type=input_content_type,
        output_content_type=output_content_type,
    )

    return output, output_content_type
