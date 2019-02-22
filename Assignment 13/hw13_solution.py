# -*- coding: utf-8 -*-
"""
Assignment 13: RNNs
@author: Aaron, Ashwin, Niral
"""

import numpy as np

class RNN:
      
    def __init__(self,new_spec):
        
        #Save spec to spec
        self.spec = new_spec.copy()
        self.nodes = {}
        
        #Parse spec and form necessary nodal storage
        all_verts = list(self.spec['vertices'])
        for vert in all_verts:
            self.nodes[vert] = {}
            self.nodes[vert]['length'] = self.spec['vertices'][vert]['num_nodes']
            self.nodes[vert]['mem'] = np.zeros(self.nodes[vert]['length'])
            self.nodes[vert]['bias'] = self.spec['vertices'][vert]['bias']
            self.nodes[vert]['func'] = self.spec['vertices'][vert]['activation']
        
        self.resetables = [vert for vert in all_verts if vert not in list(self.spec['memories'])]
    
    def activation(self,vert):
        #Activation function given some edge['source_id']
        return self.nodes[vert]['func'](self.nodes[vert]['mem'] + self.nodes[vert]['bias'])
        
    def apply(self,input_data):
        
        #Create output list with same length as data and width as output
        output_list = np.zeros((len(input_data),self.nodes[self.spec['output']]['length']))
        
        #for each data value in list
        for ind in range(len(input_data)):
            #Set out_mem->in_mem
            for mem in list(self.spec['memories']):
                self.nodes[mem]['mem'] = self.nodes[self.spec['memories'][mem]]['mem']
            #Reset other nodes
            for reset in self.resetables:
                self.nodes[reset]['mem'] = np.zeros(self.nodes[reset]['length'])
            #Set input
            self.nodes[self.spec['input']]['mem'] = input_data[ind]
        
            #for each edge, update nodes in order
            for edge in self.spec['edges']:
                act_out = self.activation(edge['source_id'])
                self.nodes[edge['target_id']]['mem'] = np.dot(act_out,edge['weights']) + self.nodes[edge['target_id']]['mem']

            #store output 
            output_list[ind] = self.activation(self.spec['output'])
        
        return output_list