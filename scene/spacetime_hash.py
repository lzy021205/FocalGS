import torch
import tinycudann as tcnn
import torch.nn as nn

class SpaceTimeHashingField(torch.nn.Module):
    def __init__(self, xyz_bound_min, xyz_bound_max , hashmap_size=16, activation = "ReLU", n_levels = 16,
        n_features_per_level=4, base_resolution = 16 ,n_neurons = 128 , feat_dim = 64  ):
        super(SpaceTimeHashingField, self).__init__()

        self.feat_dim = feat_dim

        self.enc_model = tcnn.NetworkWithInputEncoding(
            n_input_dims = 4 ,
            n_output_dims = feat_dim ,    # as same as 4dgs
            encoding_config={
                "otype": "HashGrid" ,
                "n_levels": n_levels ,
                "n_features_per_level": n_features_per_level,  # 2
                "log2_hashmap_size": hashmap_size ,
                "base_resolution": base_resolution ,
                "per_level_scale": 2.0 ,
            },

            network_config={
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": "ReLU",  
                "n_neurons": n_neurons,
                "n_hidden_layers": 1 ,
            },
        )


        self.mlp_mask = tcnn.Network(
            n_input_dims= 4,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 1,
            },
        )

        self.register_buffer('xyz_bound_min',xyz_bound_min)
        self.register_buffer('xyz_bound_max',xyz_bound_max)

    def dump(self, path):
        torch.save(self.state_dict(),path)
        

    def get_contracted_xyz(self, xyz):
        with torch.no_grad():
            contracted_xyz=(xyz-self.xyz_bound_min)/(self.xyz_bound_max-self.xyz_bound_min)
            return contracted_xyz


    def forward(self, xyz:torch.Tensor, time):
        contracted_xyz = self.get_contracted_xyz(xyz)

        mask = ( contracted_xyz >= 0 ) & ( contracted_xyz <= 1 )
        mask = mask.all(dim=1)
        
        inputs_subset = contracted_xyz[mask].float()
        time_subset = time[mask].float()
        
        hash_inputs = torch.cat([inputs_subset, time_subset], dim=-1)

        if hash_inputs.shape[0] == 0:
            dynamic_feature_out = torch.zeros((0, self.feat_dim), device=xyz.device, dtype=torch.float32)
            temp_dynamics = torch.zeros((0, 1), device=xyz.device, dtype=torch.float32)
        else:
            hash_inputs = hash_inputs.contiguous()
            dynamic_feature_out = self.enc_model(hash_inputs) 
            temp_dynamics = self.mlp_mask(hash_inputs)

        # Scatter back
        # Scatter back (dtype/device must match)
        out_dtype = dynamic_feature_out.dtype  # could be torch.float16 under amp
        device = xyz.device
        N = xyz.shape[0]

        dynamic_feature = torch.zeros((N, self.feat_dim), device=device, dtype=out_dtype)
        dynamic_mask    = torch.zeros((N, 1),            device=device, dtype=temp_dynamics.dtype)

        dynamic_feature[mask] = dynamic_feature_out
        dynamic_mask[mask]    = torch.sigmoid(temp_dynamics)


        dynamic_feature = dynamic_feature.float()
        dynamic_mask    = dynamic_mask.float()

        return dynamic_feature, dynamic_mask


    


    def get_params (self):

        parameter_list = []
        for name, param in self.named_parameters():
            parameter_list.append(param)
        return parameter_list
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters(): # enc_model.para
            if  "enc_model" not in name:
                parameter_list.append(param)
        return parameter_list


    def get_hash_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "enc_model" in name:
                parameter_list.append(param)  
        return parameter_list

        
        