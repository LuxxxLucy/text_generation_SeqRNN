import os
import keras
from keras.models import load_model

class ModelSession(object):
    """
    A OOP style object of keras

    The model's graph structure is defined by overriding the create_graph function.
    """

    def __init__(self, args):
        """
        Create a model session.

        Do not call this constructor directly. To instantiate a ModelSession object, use the create and restore class
        methods.

        :param session: the session in which this model is running
        :type session: tf.Session
        :param saver: object used to serialize this session
        :type saver: tf.Saver
        """
        self.args=args
        self.model=None

    @classmethod
    def create(cls, **kwargs):
        """
        Create a new model session.

        :param kwargs: optional graph parameters
        :type kwargs: dict
        :return: new model session
        :rtype: ModelSession
        """
        model=cls.create_graph(**kwargs)
        return cls.compile_model(model)

    @staticmethod
    def create_graph():
        """
        Create a new computation model.

        :param kwargs: optional graph parameters
        :type kwargs: dict
        :return: new model session
        :rtype: a keras Mode

        """
        raise NotImplementedError

    @classmethod
    def compile_model(cls, model):
        """
        compile current model, cost and optimizer

        :param kwargs: model
        :type kwargs: a Keras Model
        :return: new model session
        :rtype: a keras Model

        """
        raise NotImplementedError

    def restore(self, checkpoint_directory,cus=None):
        """
        Restore a serialized model session.

        :param checkpoint_directory:  directory containing checkpoint files
        :type checkpoint_directory: str
        :return: model restored from the latest checkpoint file
        :rtype: keras model
        """

        path=os.path.join(checkpoint_directory,'model.h5')
        self.model = load_model(path,custom_objects=cus)
        return self.compile_model(model)


    def save(self, checkpoint_directory):
        """
        Save the current model session to a checkpoint file.

        If the graph defines an "iteration" variable its value will be used for the global step in the checkpoint name.

        :param checkpoint_directory:  directory containing checkpoint files
        :type checkpoint_directory: str
        :return: path to the new checkpoint file
        :rtype: str
        """

        path=os.path.join(checkpoint_directory,'model.h5')

        try:
            self.model.save(path)
            return "save okay!"
        except:
            return "Fatal error! Saving failed"

    def summary(self):
        self.model.summary()
