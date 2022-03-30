"""
Model construction utilities based on keras
"""
import warnings
from distutils.version import LooseVersion
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten

# from cleverhans.model import Model, NoSuchLayerError
import tensorflow as tf

from abc import ABCMeta


class NoSuchLayerError(ValueError):
    """Raised when a layer that does not exist is requested."""

class AModel(object):
    """
    An abstract interface for model wrappers that exposes model symbols
    needed for making an attack. This abstraction removes the dependency on
    any specific neural network package (e.g. Keras) from the core
    code of CleverHans. It can also simplify exposing the hidden features of a
    model when a specific package does not directly expose them.
    """

    __metaclass__ = ABCMeta
    O_LOGITS, O_PROBS, O_FEATURES = "logits probs features".split()

    def __init__(
        self, scope=None, nb_classes=None, hparams=None, needs_dummy_fprop=False
    ):
        """
        Constructor.
        :param scope: str, the name of model.
        :param nb_classes: integer, the number of classes.
        :param hparams: dict, hyper-parameters for the model.
        :needs_dummy_fprop: bool, if True the model's parameters are not
            created until fprop is called.
        """
        self.scope = scope or self.__class__.__name__
        self.nb_classes = nb_classes
        self.hparams = hparams or {}
        self.needs_dummy_fprop = needs_dummy_fprop

    def __call__(self, *args, **kwargs):
        """
        For compatibility with functions used as model definitions (taking
        an input tensor and returning the tensor giving the output
        of the model on that input).
        """

        warnings.warn(
            "Model.__call__ is deprecated. "
            "The call is ambiguous as to whether the output should "
            "be logits or probabilities, and getting the wrong one "
            "can cause serious problems. "
            "The output actually is probabilities, which are a very "
            "dangerous thing to use as part of any interface for "
            "cleverhans, because softmax probabilities are prone "
            "to gradient masking."
            "On or after 2019-04-24, this method will change to raise "
            "an exception explaining why Model.__call__ should not be "
            "used."
        )

        return self.get_probs(*args, **kwargs)

    def get_logits(self, x, **kwargs):
        """
        :param x: A symbolic representation (Tensor) of the network input
        :return: A symbolic representation (Tensor) of the output logits
        (i.e., the values fed as inputs to the softmax layer).
        """
        outputs = self.fprop(x, **kwargs)
        if self.O_LOGITS in outputs:
            return outputs[self.O_LOGITS]
        raise NotImplementedError(
            str(type(self)) + "must implement `get_logits`"
            " or must define a " + self.O_LOGITS + " output in `fprop`"
        )

    def get_predicted_class(self, x, **kwargs):
        """
        :param x: A symbolic representation (Tensor) of the network input
        :return: A symbolic representation (Tensor) of the predicted label
        """
        return tf.argmax(self.get_logits(x, **kwargs), axis=1)

    def get_probs(self, x, **kwargs):
        """
        :param x: A symbolic representation (Tensor) of the network input
        :return: A symbolic representation (Tensor) of the output
        probabilities (i.e., the output values produced by the softmax layer).
        """
        d = self.fprop(x, **kwargs)
        if self.O_PROBS in d:
            output = d[self.O_PROBS]
            min_prob = tf.reduce_min(output)
            max_prob = tf.reduce_max(output)
            asserts = [
                utils_tf.assert_greater_equal(min_prob, tf.cast(0.0, min_prob.dtype)),
                utils_tf.assert_less_equal(max_prob, tf.cast(1.0, min_prob.dtype)),
            ]
            with tf.control_dependencies(asserts):
                output = tf.identity(output)
            return output
        elif self.O_LOGITS in d:
            return tf.nn.softmax(logits=d[self.O_LOGITS])
        else:
            raise ValueError("Cannot find probs or logits.")

    def fprop(self, x, **kwargs):
        """
        Forward propagation to compute the model outputs.
        :param x: A symbolic representation of the network input
        :return: A dictionary mapping layer names to the symbolic
                 representation of their output.
        """
        raise NotImplementedError("`fprop` not implemented.")

    def get_params(self):
        """
        Provides access to the model's parameters.
        :return: A list of all Variables defining the model parameters.
        """

        if hasattr(self, "params"):
            return list(self.params)

        # Catch eager execution and assert function overload.
        try:
            if tf.executing_eagerly():
                raise NotImplementedError(
                    "For Eager execution - get_params " "must be overridden."
                )
        except AttributeError:
            pass

        # For graph-based execution
        scope_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + "/"
        )

        if len(scope_vars) == 0:
            self.make_params()
            scope_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + "/"
            )
            assert len(scope_vars) > 0

        # Make sure no parameters have been added or removed
        if hasattr(self, "num_params"):
            if self.num_params != len(scope_vars):
                print("Scope: ", self.scope)
                print("Expected " + str(self.num_params) + " variables")
                print("Got " + str(len(scope_vars)))
                for var in scope_vars:
                    print("\t" + str(var))
                assert False
        else:
            self.num_params = len(scope_vars)

        return scope_vars

    def make_params(self):
        """
        Create all Variables to be returned later by get_params.
        By default this is a no-op.
        Models that need their fprop to be called for their params to be
        created can set `needs_dummy_fprop=True` in the constructor.
        """

        if self.needs_dummy_fprop:
            if hasattr(self, "_dummy_input"):
                return
            self._dummy_input = self.make_input_placeholder()
            self.fprop(self._dummy_input)

    def get_layer_names(self):
        """Return the list of exposed layers for this model."""
        raise NotImplementedError

    def get_layer(self, x, layer, **kwargs):
        """Return a layer output.
        :param x: tensor, the input to the network.
        :param layer: str, the name of the layer to compute.
        :param **kwargs: dict, extra optional params to pass to self.fprop.
        :return: the content of layer `layer`
        """
        return self.fprop(x, **kwargs)[layer]

    def make_input_placeholder(self):
        """Create and return a placeholder representing an input to the model.
        This method should respect context managers (e.g. "with tf.device")
        and should not just return a reference to a single pre-created
        placeholder.
        """

        raise NotImplementedError(
            str(type(self)) + " does not implement " "make_input_placeholder"
        )

    def make_label_placeholder(self):
        """Create and return a placeholder representing class labels.
        This method should respect context managers (e.g. "with tf.device")
        and should not just return a reference to a single pre-created
        placeholder.
        """

        raise NotImplementedError(
            str(type(self)) + " does not implement " "make_label_placeholder"
        )

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other


class KerasModelWrapper(AModel):
    """
  An implementation of `Model` that wraps a Keras model. It
  specifically exposes the hidden features of a model by creating new models.
  The symbolic graph is reused and so there is little overhead. Splitting
  in-place operations can incur an overhead.
  """

    def __init__(self, model, num_class=10):
        """
        Create a wrapper for a Keras model
        :param model: A Keras model
        """
        super(KerasModelWrapper, self).__init__()

        if model is None:
            raise ValueError('model argument must be supplied.')

        self.model = model
        self.keras_model = None
        self.num_classes = num_class

    def _get_softmax_name(self):
        """
    Looks for the name of the softmax layer.
    :return: Softmax layer name
    """
        for layer in self.model.layers:
            cfg = layer.get_config()
            if cfg['name'] == 'average_1':
                return layer.name

        raise Exception("No softmax layers found")

    def _get_logits_name(self):
        """
    Looks for the name of the layer producing the logits.
    :return: name of layer producing the logits
    """
        softmax_name = self._get_softmax_name()
        softmax_layer = self.model.get_layer(softmax_name)

        if not isinstance(softmax_layer, Activation):
            # In this case, the activation is part of another layer
            return softmax_name

        if hasattr(softmax_layer, 'inbound_nodes'):
            warnings.warn(
                "Please update your version to keras >= 2.1.3; "
                "support for earlier keras versions will be dropped on "
                "2018-07-22")
            node = softmax_layer.inbound_nodes[0]
        else:
            node = softmax_layer._inbound_nodes[0]

        logits_name = node.inbound_layers[0].name

        return logits_name

    def get_logits(self, x):
        """
    :param x: A symbolic representation of the network input.
    :return: A symbolic representation of the logits
    """
        # logits_name = self._get_logits_name()
        # logits_layer = self.get_layer(x, logits_name)

        # # Need to deal with the case where softmax is part of the
        # # logits layer
        # if logits_name == self._get_softmax_name():
        #   softmax_logit_layer = self.get_layer(x, logits_name)

        #   # The final op is the softmax. Return its input
        #   logits_layer = softmax_logit_layer._op.inputs[0]
        prob = self.get_probs(x)
        logits = tf.log(prob)

        return logits

    def get_probs(self, x):
        """
    :param x: A symbolic representation of the network input.
    :return: A symbolic representation of the probs
    """

        return self.model(x)

    def get_layer_names(self):
        """
    :return: Names of all the layers kept by Keras
    """
        layer_names = [x.name for x in self.model.layers]
        return layer_names

    def fprop(self, x):
        """
    Exposes all the layers of the model returned by get_layer_names.
    :param x: A symbolic representation of the network input
    :return: A dictionary mapping layer names to the symbolic
             representation of their output.
    """
        from tensorflow.keras.models import Model as KerasModel

        if self.keras_model is None:
            # Get the input layer
            new_input = self.model.get_input_at(0)

            # Make a new model that returns each of the layers as output
            out_layers = [x_layer.output for x_layer in self.model.layers]
            self.keras_model = KerasModel(new_input, out_layers)

        # and get the outputs for that model on the input x
        outputs = self.keras_model(x)

        # Keras only returns a list for outputs of length >= 1, if the model
        # is only one layer, wrap a list
        if len(self.model.layers) == 1:
            outputs = [outputs]

        # compute the dict to return
        fprop_dict = dict(zip(self.get_layer_names(), outputs))

        return fprop_dict

    def get_layer(self, x, layer):
        """
    Expose the hidden features of a model given a layer name.
    :param x: A symbolic representation of the network input
    :param layer: The name of the hidden layer to return features at.
    :return: A symbolic representation of the hidden features
    :raise: NoSuchLayerError if `layer` is not in the model.
    """
        # Return the symbolic representation for this layer.
        output = self.fprop(x)
        try:
            requested = output[layer]
        except KeyError:
            raise NoSuchLayerError()
        return requested
