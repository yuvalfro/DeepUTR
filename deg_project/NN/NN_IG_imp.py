from deg_project.general.utilies import one_hot_encoding
import random
import numpy as np
import tensorflow as tf

from deg_project.general import utilies

from tqdm import tqdm

def get_gradients(model, sample_inputs, target_range=None, jacobian=False):
    """Computes the gradients of outputs w.r.t input.

    Args:
        sample_inputs (ndarray):: model sample inputs
        target_rtarget_range (slice)ange: Range of target

    Returns:
        Gradients of the predictions w.r.t input
    """
    combine = np.dstack((sample_inputs[0], sample_inputs[1]))
    sample_inputs = [combine, sample_inputs[2]]
    if isinstance(sample_inputs, list):
        for i in range(len(sample_inputs)):
            sample_inputs[i] = tf.cast(sample_inputs[i], tf.float32)
    else:
        sample_inputs = tf.cast(sample_inputs, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(sample_inputs)
        preds = model(sample_inputs)
        if (target_range is None):
            target_preds = preds[:, :]
        else:
            target_preds = preds[:, target_range]

    if(jacobian):
        grads = tape.jacobian(target_preds, sample_inputs)
    else:
        grads = tape.gradient(target_preds, sample_inputs)

    if isinstance(grads, list):
        grads = [grad.numpy() for grad in grads]
        one_hot = grads[0][:, :, 0:4]
        lunps = grads[0][:, :, 4:5]
        grads = [one_hot, lunps, grads[1]]
    else:
        grads = grads.numpy()
    return grads


def linearly_interpolate(sample_input, baseline=None, num_steps=50, multiple_samples=False, const=False):
    # If baseline is not provided, start with a zero baseline
    # having same size as the sample input.
    if baseline is None:
        baseline = np.zeros(sample_input.shape).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    # Do interpolation.
    sample_input = sample_input.astype(np.float32)
    if(const):
        #just duplicate the same sample_input value num_steps + 1 times without the linear interpolation
        interpolated_sample_input = [
            sample_input for _ in range(num_steps + 1)]
    else:
        interpolated_sample_input = [
            baseline + (step / num_steps) * (sample_input - baseline)
            for step in range(num_steps + 1)
        ]

    interpolated_sample_input = np.array(
        interpolated_sample_input).astype(np.float32)
    if(multiple_samples):
        old_shape = interpolated_sample_input.shape
        # switch the two first axises
        new_transpose_form = (
            1, 0)+tuple(range(interpolated_sample_input.ndim)[2:])
        new_shape_form = (old_shape[0]*old_shape[1],) + old_shape[2:]
        interpolated_sample_input = interpolated_sample_input.transpose(
            new_transpose_form).reshape(new_shape_form)

    return interpolated_sample_input, sample_input, baseline


def get_integrated_gradients(model, sample_inputs, target_range=None, baselines=None, num_steps=50, multiple_samples=False, batch_size=128, const_inputs=None):
    """Computes Integrated Gradients for range of labels.

    Args:
        model (tensorflow model): Model
        sample_inputs (ndarray): Original sample input to the model
        target_range (slice): Target range - grdient of Target range  with respect to the input
        baseline (ndarray): The baseline to start with for interpolation
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients. These
            steps along determine the integral approximation error. By default,
            num_steps is set to 50.

    Returns:
        integrated_grads_list : Integrated gradients w.r.t input
        hypothetical_importance_list : hypothetical importance w.r.t input
    """
    # tf.compat.v1.enable_v2_behavior()
    if isinstance(sample_inputs, list):
        inputs_types_num = len(sample_inputs)
    else:
        # insert the inputs to a list to fit generalized code
        inputs_types_num = 1
        sample_inputs = [sample_inputs]
        if (baselines is not None):
            baselines = [baselines]

    # 1. Do interpolation.
    output = []
    if(baselines is None):
        for i, sample_input in enumerate(sample_inputs):
            if(const_inputs is not None):
                const = True if (i in const_inputs) else False
            else:
                const = False
            output.append(linearly_interpolate(sample_input, baselines,
                          num_steps=num_steps, multiple_samples=multiple_samples, const=const))
    else:
        for i, (sample_input, baseline) in enumerate(zip(sample_inputs, baselines)):
            if(const_inputs is not None):
                const = True if (i in const_inputs) else False
            else:
                const = False
            output.append(linearly_interpolate(sample_input, baseline,
                          num_steps=num_steps, multiple_samples=multiple_samples, const=const))

    interpolated_samples_inputs = [x[0] for x in output]
    sample_inputs = [x[1] for x in output]
    baselines = [x[2] for x in output]

    # 2. Get the gradients
    interpolated_samples_num = len(interpolated_samples_inputs[0])
    if (inputs_types_num == 1):
        grads_list = [np.concatenate([get_gradients(model, interpolated_samples_inputs[0][i:i+batch_size],
                                 target_range=target_range) for i in tqdm(range(0, interpolated_samples_num, batch_size), desc="IG progress")], 0)]
    else:
        # create list of batch size grands lists
        grads_list = [get_gradients(model, [interpolated_samples_input[i:i+batch_size] for interpolated_samples_input in interpolated_samples_inputs],
                                    target_range=target_range) for i in tqdm(range(0, interpolated_samples_num, batch_size), desc="IG progress")]
        # Concatenate the batch size lists
        grads_list = [np.concatenate([grads_list_element[i] for grads_list_element in grads_list], axis=0)
                      for i in range(inputs_types_num)]

    if(multiple_samples):
        num_of_samples = sample_inputs[0].shape[0]
        grads_list = [np.reshape(grads, (num_of_samples, num_steps+1) + grads.shape[1:]) for grads in grads_list]

    # 3. Approximate the integral using the trapezoidal rule
    if(multiple_samples):
        grads_list = [(grads[:, :-1] + grads[:, 1:]) /
                      2.0 for grads in grads_list]
        avg_grads_list = [np.mean(grads, axis=1)
                          for grads in grads_list]
    else:
        grads_list = [(grads[:-1] + grads[1:]) / 2.0 for grads in grads_list]
        avg_grads_list = [np.mean(grads, axis=0)
                          for grads in grads_list]

    # 4. get hypothetical importance score - it's the average gradient
    hypothetical_importance_list = avg_grads_list

    # 5. Calculate integrated gradients and return
    integrated_grads_list = [(sample_inputs[i] - baselines[i])
                             * avg_grads_list[i] for i in range(inputs_types_num)]
    if (inputs_types_num == 1):
        return integrated_grads_list[0], hypothetical_importance_list[0]
    return integrated_grads_list, hypothetical_importance_list


def get_integrated_gradients_random(model, sample_inputs, target_range=None, baselines=None, num_steps=50, multiple_samples=False, batch_size=128, baseline_num=100, const_inputs=None):
    # this works for 8_points, fold_change and linear models
    inputs_types_num = len(sample_inputs)
    seq_len = 110
    baseline_seqs = [''.join(random.choices("ACGT", k=seq_len))
                     for i in range(baseline_num)]
    (integrated_grads, hypothetical_importance) = ((np.zeros(sample_inputs[0].shape), np.zeros(sample_inputs[1].shape)), (np.zeros(
        sample_inputs[0].shape), np.zeros(sample_inputs[1].shape))) if inputs_types_num == 2 else (np.zeros(sample_inputs.shape), np.zeros(sample_inputs.shape))
    for i in range(baseline_num):
        seq_basline = one_hot_encoding(baseline_seqs[i])
        baselines = [seq_basline, None if baselines is None else baselines[1]
                     ] if inputs_types_num == 2 else seq_basline
        integrated_grads_list, hypothetical_importance_list = get_integrated_gradients(
            model, sample_inputs, target_range=target_range, baselines=baselines, num_steps=num_steps, multiple_samples=multiple_samples, batch_size=batch_size, const_inputs=const_inputs)

        (integrated_grads, hypothetical_importance) = ((integrated_grads[0]+integrated_grads_list[0]/baseline_num, integrated_grads[1]+integrated_grads_list[1]/baseline_num), (hypothetical_importance[0]+hypothetical_importance_list[0] /
                                                                                                                                                                              baseline_num, hypothetical_importance[1]+hypothetical_importance_list[1]/baseline_num)) if inputs_types_num == 2 else (integrated_grads+integrated_grads_list/baseline_num, hypothetical_importance+hypothetical_importance_list/baseline_num)

    integrated_grads = [sample_inputs[0]*hypothetical_importance[0], sample_inputs[1] *
                        hypothetical_importance[1]] if inputs_types_num == 2 else sample_inputs*hypothetical_importance
    hypothetical_importance = [hypothetical_importance[0], hypothetical_importance[1]
                               ] if inputs_types_num == 2 else hypothetical_importance

    return integrated_grads, hypothetical_importance


def call_ig(model, sample_inputs, target_range=None, multiple_samples=False, interpretability_args=None):
    if(interpretability_args is None):
        return get_integrated_gradients(model, sample_inputs, target_range, multiple_samples=multiple_samples)

    default_args = utilies.get_default_args(get_integrated_gradients)
    baselines = interpretability_args['baselines'] if 'baselines' in interpretability_args else default_args['baselines']
    num_steps = interpretability_args['num_steps'] if 'num_steps' in interpretability_args else default_args['num_steps']
    batch_size = interpretability_args['batch_size'] if 'batch_size' in interpretability_args else default_args['batch_size']
    const_inputs = interpretability_args['const_inputs'] if 'const_inputs' in interpretability_args else default_args['const_inputs']
    if(interpretability_args['type'] == 'regular'):
        return get_integrated_gradients(model, sample_inputs, target_range, baselines=baselines, num_steps=num_steps, multiple_samples=multiple_samples, batch_size=batch_size, const_inputs=const_inputs)
    elif(interpretability_args['type'] == 'random'):
        default_args = utilies.get_default_args(
            get_integrated_gradients_random)
        baseline_num = interpretability_args['baseline_num'] if 'baseline_num' in interpretability_args else default_args['baseline_num']
        return get_integrated_gradients_random(model, sample_inputs, target_range=target_range, baselines=baselines, num_steps=num_steps, multiple_samples=multiple_samples, batch_size=batch_size, baseline_num=baseline_num, const_inputs=const_inputs)
    else:
        raise ValueError('invalid IG type')
