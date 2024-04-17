# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import net


class DygraphModel():
    def create_model(self, config):
        feature_vocabulary = config.get("hyper_parameters.feature_vocabulary")
        embedding_size = config.get("hyper_parameters.embedding_size")
        tower_dims = config.get("hyper_parameters.dims")
        drop_prob = config.get('hyper_parameters.drop_prob')
        feature_vocabulary = dict(feature_vocabulary)
        model = net.AITM(feature_vocabulary, embedding_size, tower_dims,
                         drop_prob)
        return model

    # define feeds which convert numpy of batch data to paddle.tensor 
    def create_feeds(self, batch_data, config):
        click, conversion, features = batch_data
        return click.astype('float32'), conversion.astype('float32'), features

    # define loss function by predicts and label
    def create_loss(self,
                    ctr_pred,
                    ctcvr_pred,
                    click_label,
                    conversion_label,
                    cvr_pred,
                    side_cvr_pred,
                    constraint_weight=0.6):
        ctr_loss = F.binary_cross_entropy(ctr_pred, click_label)
        ctcvr_loss = F.binary_cross_entropy(ctcvr_pred, conversion_label,reduction='none')
        ctcvr_loss = paddle.multiply(ctcvr_loss,click_label)
        ctcvr_loss += F.binary_cross_entropy(paddle.multiply(ctr_pred,cvr_pred),conversion_label,reduction='none')
        ctcvr_loss = paddle.mean(ctcvr_loss)

        ctcvr_pred_max = ctcvr_pred
        cconversion_label = conversion_label+ctcvr_pred_max
        cconversion_label = paddle.minimum(cconversion_label,paddle.full_like(cconversion_label,1))
        
        cconversion_label.stop_gradient = True
        cvr_loss = F.binary_cross_entropy(cvr_pred, cconversion_label,reduction='none')
        O=paddle.cast(click_label, 'float32')
        cvr_loss = paddle.multiply(cvr_loss,O)
        min_v = paddle.full_like(ctr_pred, 0.000001)
        PS = paddle.maximum(ctr_pred, min_v)
        IPS = paddle.reciprocal(PS)

        if paddle.sum(O) > 0:
            IPS = IPS/paddle.sum(paddle.multiply(IPS,O))
        IPS.stop_gradient = True
        cvr_loss = paddle.multiply(cvr_loss, IPS)

        cvr_loss = paddle.mean(cvr_loss)


        cconversion_label2 = ctcvr_pred
        cconversion_label2.stop_gradient = True
        side_cvr_loss = F.binary_cross_entropy(cvr_pred, cconversion_label2,reduction='none')

        side_cvr_loss = paddle.multiply(side_cvr_loss,1-O)
        side_PS = paddle.maximum(1-ctr_pred, min_v)
        side_IPS = paddle.reciprocal(side_PS)
        if paddle.sum(1-O) > 0:
            side_IPS = side_IPS/paddle.sum(paddle.multiply(side_IPS,1-O))
        side_IPS.stop_gradient = True
        side_cvr_loss = paddle.multiply(side_cvr_loss, side_IPS)
        side_cvr_loss = paddle.mean(side_cvr_loss)

        cvr_loss = cvr_loss + side_cvr_loss
        

        loss = ctr_loss + ctcvr_loss + cvr_loss
        return loss

    # define optimizer 
    def create_optimizer(self, dy_model, config):
        lr = config.get("hyper_parameters.optimizer.learning_rate", 0.0001)
        optimizer = paddle.optimizer.Adam(
            learning_rate=lr,
            parameters=dy_model.parameters(),
            weight_decay=1e-6)
        return optimizer

    # define metrics such as auc/acc
    # multi-task need to define multi metric
    def create_metrics(self):
        metrics_list_name = ["ctr_auc", "cvr_auc", "ctcvr"]
        metrics_list = [
            paddle.metric.Auc("ROC", num_thresholds=100000),
            paddle.metric.Auc("ROC", num_thresholds=100000),
            paddle.metric.Auc("ROC", num_thresholds=100000)
        ]
        return metrics_list, metrics_list_name

    # construct train forward phase  
    def train_forward(self, dy_model, metrics_list, batch_data, config):
        click, conversion, features = self.create_feeds(batch_data, config)
        ctr_pred, ctcvr_pred, cvr_pred, imputation = dy_model.forward(features)
        loss = self.create_loss(ctr_pred, ctcvr_pred, click, conversion,cvr_pred,imputation)
        # update metrics

        self.update_auc(ctr_pred, click, metrics_list[0])
        if paddle.sum(click) > 0:
            ccvr_pred = paddle.gather(cvr_pred, paddle.where(click == 1))
            cconversion = paddle.gather(conversion, paddle.where(click == 1))
            self.update_auc(ccvr_pred, cconversion, metrics_list[1])

        self.update_auc(paddle.multiply(ctr_pred,cvr_pred), conversion, metrics_list[2])
        print_dict = {'loss': loss}
        return loss, metrics_list, print_dict

    @staticmethod
    def update_auc(prob, label, metrics):
        if prob.ndim == 1:
            prob = prob.unsqueeze(-1)
        assert prob.ndim == 2
        predict_2d = paddle.concat(x=[1 - prob, prob], axis=1)
        label = label.reshape([-1,1])
        metrics.update(predict_2d, label)

    def infer_forward(self, dy_model, metrics_list, batch_data, config):
        click, conversion, features = self.create_feeds(batch_data, config)
        with paddle.no_grad():
            ctr_pred, ctcvr_pred, cvr_pred, imputation = dy_model.forward(features)
        # update metrics
        self.update_auc(ctr_pred, click, metrics_list[0])
        if paddle.sum(click) > 0:
            ccvr_pred = paddle.gather(cvr_pred, paddle.where(click == 1))
            cconversion = paddle.gather(conversion, paddle.where(click == 1))
            self.update_auc(ccvr_pred, cconversion, metrics_list[1])
        self.update_auc(paddle.multiply(ctr_pred,cvr_pred), conversion, metrics_list[2])
        return metrics_list, None

    def forward(self, dy_model, batch_data, config):
        click, conversion, features = self.create_feeds(batch_data, config)
        with paddle.no_grad():
            click_pred, conversion_pred = dy_model.forward(features)
        # update metrics
        return click, click_pred, conversion, conversion_pred
    

    def infer_forward_out(self, dy_model, metrics_list, batch_data, config):
        click, conversion, features = self.create_feeds(batch_data, config)
        with paddle.no_grad():
            ctr_pred, ctcvr_pred, cvr_pred, imputation = dy_model.forward(features)
        
        results = [ctr_pred.reshape([-1]).numpy(), cvr_pred.reshape([-1]).numpy(), ctcvr_pred.reshape([-1]).numpy()]
        labels = [click.reshape([-1]).numpy(), conversion.reshape([-1]).numpy()]
        # update metrics
        self.update_auc(ctr_pred, click, metrics_list[0])
        if paddle.sum(click) > 0:
            cvr_pred = paddle.gather(cvr_pred, paddle.where(click == 1))
            cconversion = paddle.gather(conversion, paddle.where(click == 1))
            self.update_auc(cvr_pred, cconversion, metrics_list[1])
        self.update_auc(ctcvr_pred, conversion, metrics_list[2])
               

        return metrics_list, None,results,labels
