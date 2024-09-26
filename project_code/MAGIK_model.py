from deeplay import (
             Parallel, FromDict, MultiLayerPerceptron, DeeplayModule,
             LearnableDistancewWeighting, WeightedSum, Layer, LayerList,
             Transform, TransformPropagateUpdate, Update, Cat, Sum, GraphToEdgeMAGIK
            )
import deeplay as dl
from typing import List, Optional, Literal, Any, Sequence, Type, overload, Union
import torch.nn as nn


class MessagePassingNeuralNetwork(DeeplayModule):
    @property
    def input(self):
      return self.blocks[0]

    @property
    def hidden(self):
      return self.blocks[:-1]

    @property
    def output(self):
        return self.blocks[-1]

    @property
    def transform(self) -> LayerList[Layer]:
        return self.blocks.transform

    @property
    def propagate(self) -> LayerList[Layer]:
        return self.blocks.propagate

    @property
    def update(self) -> LayerList[Layer]:
        return self.blocks.update

    def __init__(self, hidden_features, out_features, out_activation: Union[Type[nn.Module], nn.Module, None] = None):
        super().__init__()
        self.hidden_features = hidden_features
        self.out_features = out_features
        out_activation = Layer(out_activation)
        self.blocks = LayerList()
        for i, c_out in enumerate([*hidden_features, out_features]):
            activation = (
                Layer(nn.ReLU) if i < len(hidden_features) - 1 else out_activation
            )

            transform = Transform(
                combine=Cat(),
                layer=Layer(nn.LazyLinear, c_out),
                activation=activation.new(),
            )
            transform.set_input_map("x", "edge_index", "edge_attr")
            transform.set_output_map("edge_attr")

            propagate = Sum()
            propagate.set_input_map("x", "edge_index", "edge_attr")
            propagate.set_output_map("aggregate")

            update = Update(
                combine=Cat(),
                layer=Layer(nn.LazyLinear, c_out),
                activation=activation.new(),
            )
            update.set_input_map("x", "aggregate")
            update.set_output_map("x")

            block = TransformPropagateUpdate(
                transform=transform,
                propagate=propagate,
                update=update,
            )

            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Classifier_model(DeeplayModule):
  def __init__(self):
    super().__init__()
    self.encoder = Parallel(
        **{ key: MultiLayerPerceptron(
                    in_features=None,
                    hidden_features=[32, 64],
                    out_features=96,
                    flatten_input=False,
                ).set_input_map(key)
         for key in ["x", "edge_attr"]
        }
    )

    self.mpm = MessagePassingNeuralNetwork(
        hidden_features=[96]*3,
        out_features= 96,
        out_activation=nn.ReLU,
    )
    distance_embedder = LearnableDistancewWeighting()
    distance_embedder.set_input_map("distance")
    distance_embedder.set_output_map("edge_weight")
    self.mpm.blocks.insert(0, distance_embedder)
    propagate = WeightedSum()
    propagate.set_input_map("x", "edge_index", "edge_attr", "edge_weight")
    propagate.set_output_map("aggregate")

    for block in self.mpm.blocks[1:]:
      block.replace("propagate", propagate.new())

    self.selector = FromDict("edge_attr")

    self.pool = nn.Identity()

    self.head = nn.Sequential(
        nn.LazyLinear(out_features=64, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=1, bias=True),
        nn.Sigmoid(),
    )

  def forward(self, graph):
    graph = self.encoder(graph)
    graph = self.mpm(graph)
    graph = self.selector(graph)
    graph = self.pool(graph)
    graph = self.head(graph)
    return graph