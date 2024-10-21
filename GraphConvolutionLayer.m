classdef GraphConvolutionLayer < nnet.layer.Layer
    properties
        NumNodes
        NumFeatures
        Weights
    end
    
    methods
        function layer = GraphConvolutionLayer(numNodes, numFeatures, name)
            layer.Name = name;
            layer.NumNodes = numNodes;
            layer.NumFeatures = numFeatures;
            layer.Weights = rand(numFeatures, numFeatures); % Initialize weights
        end
        
        function Z = predict(layer, X, A)
            Z = A * X * layer.Weights;
        end
    end
end
