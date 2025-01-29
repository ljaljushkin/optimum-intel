#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from nncf.torch import load_from_config

class FQLoRAModel:
    def save_pretrained(self):
        nncf_state_dict = wrapped_model.nncf.state_dict()
        nncf_config = wrapped_model.nncf.get_config()

    def from_pretrained(self):
        nncf_model = load_from_config(model, nncf_ckpt["nncf_config"], example_input=example_input)
        nncf_model.nncf.load_state_dict(nncf_ckpt["nncf_state_dict"])
