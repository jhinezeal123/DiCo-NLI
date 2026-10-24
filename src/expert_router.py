"""Candidate expert gate. Fit only inside a tracked experiment, never from dev labels.

The staged prototype is not a measured solution until a tracked run executes it.
"""
import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REV = np.array([2, 1, 0, 3])


def normalize(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1)
    return p / p.sum(1, keepdims=True)


def teacher_probabilities(evidence):
    ef, eb = evidence[:, 0], evidence[:, 3]
    return normalize(np.c_[(1-ef)*eb, ef*eb, ef*(1-eb), (1-ef)*(1-eb)])


def canonical_texts(texts):
    return [" [PAIR] ".join(sorted(" ".join(t.lower().split()) for t in pair)) for pair in texts]


def compose_negative_gate(p,gate):
    p=normalize(p)
    reversible=p[:,:3]/p[:,:3].sum(1,keepdims=True)
    return np.c_[(1-gate[:,None])*reversible,gate]


def numeric_features(texts, p, evidence):
    p = normalize(p)
    t = teacher_probabilities(evidence)
    def summary(pr):
        ordered = np.sort(pr, axis=1)
        return np.c_[pr[:, 1], pr[:, 3], pr[:, 0]+pr[:, 2], abs(pr[:, 0]-pr[:, 2]),
                     ordered[:, -1], ordered[:, -1]-ordered[:, -2], -(pr*np.log(pr)).sum(1)]
    evidence_symmetric = np.c_[np.minimum(evidence[:, :3], evidence[:, 3:]),
                               np.maximum(evidence[:, :3], evidence[:, 3:])]
    cross = np.c_[(p*t).sum(1), ((p-t)**2).sum(1), p[:, 1]*t[:, 3], p[:, 3]*t[:, 1],
                  (p[:, 2]-p[:, 0])*(t[:, 2]-t[:, 0])]
    lex = []
    for a, b in texts:
        sa, sb = set(re.findall(r"\w+",a.lower())), set(re.findall(r"\w+",b.lower()))
        na, nb = len(a.split()), len(b.split())
        lex.append([np.log1p(min(na,nb)),np.log1p(max(na,nb)),abs(na-nb)/max(na,nb,1),
                    len(sa&sb)/max(len(sa|sb),1),float(sa<=sb or sb<=sa),float(not(sa&sb))])
    return np.c_[summary(p), summary(t), evidence_symmetric, cross, np.asarray(lex)]


class ExpertRouter:
    def __init__(self, kind, text_components=16, policy="expert", use_evidence=True):
        self.kind, self.text_components = kind, text_components
        self.policy, self.use_evidence = policy, use_evidence

    def _evidence(self,evidence):
        if self.use_evidence:
            return evidence
        return np.tile([0.5,0.25,0.25,0.5,0.25,0.25],(len(evidence),1))

    def _features(self, texts, p, evidence):
        evidence = self._evidence(evidence)
        numeric = numeric_features(texts, p, evidence)
        topic = self.svd.transform(self.vectorizer.transform(canonical_texts(texts)))
        t = teacher_probabilities(evidence)
        roles = np.c_[p[:,1],p[:,3],p[:,0]+p[:,2],t[:,1],t[:,3],t[:,0]+t[:,2]]
        return np.c_[numeric,topic,(topic[:,:,None]*roles[:,None,:]).reshape(len(p),-1)]

    def fit(self, texts, p, evidence, labels):
        p = normalize(p)
        teacher = teacher_probabilities(evidence)
        self.vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=2,sublinear_tf=True)
        tokens = self.vectorizer.fit_transform(canonical_texts(texts))
        self.svd = TruncatedSVD(n_components=self.text_components,random_state=20261024)
        self.svd.fit(tokens)
        base_correct = p.argmax(1)==labels
        teacher_correct = teacher.argmax(1)==labels
        if self.policy=="negative_gate":
            learn = np.ones(len(labels),dtype=bool)
            target = (labels==3).astype(int)
        elif self.policy=="expert":
            learn = base_correct ^ teacher_correct
            target = teacher_correct[learn].astype(int)
        else:
            raise ValueError(self.policy)
        assert learn.sum()>100 and set(target)=={0,1}
        features = self._features(texts,p,evidence)
        if self.kind=="linear":
            self.model = make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=1000,random_state=20261024))
        elif self.kind=="boosted":
            self.model = HistGradientBoostingClassifier(learning_rate=0.05,max_iter=60,max_leaf_nodes=7,
                         min_samples_leaf=30,l2_regularization=10,max_bins=63,early_stopping=False,random_state=20261024)
        else:
            raise ValueError(self.kind)
        self.model.fit(features[learn],target)
        self.diagnostics={"train_rows":len(texts),"routing_supervision_rows":int(learn.sum()),
                          "positive_binary_targets":int(target.sum()),"negative_binary_targets":int((1-target).sum()),
                          "features":features.shape[1],"vocabulary":len(self.vectorizer.vocabulary_),
                          "topic_components":self.text_components,
                          "decision_rule":"chain_rule_argmax" if self.policy=="negative_gate" else "gate_threshold_0.5",
                          "policy":self.policy,"use_evidence":self.use_evidence}
        self.diagnostics["learner_parameters"]={k:v for k,v in self.model.get_params().items()
             if isinstance(v,(str,int,float,bool,type(None)))}
        return self

    def predict(self,texts,p,evidence):
        p = normalize(p)
        features = self._features(texts,p,evidence)
        reverse = self._features([(b,a) for a,b in texts],p[:,REV],evidence[:,[3,4,5,0,1,2]])
        assert np.allclose(features,reverse,rtol=1e-10,atol=1e-10), "Routing features must be reversal invariant"
        gate = self.model.predict_proba(features)[:,1]
        if self.policy=="negative_gate":
            joint=compose_negative_gate(p,gate)
            np.testing.assert_allclose(compose_negative_gate(p,p[:,3]),p,atol=1e-12)
            np.testing.assert_allclose(joint.sum(1),1,atol=1e-12)
            pred = joint.argmax(1)
        else:
            teacher = teacher_probabilities(evidence)
            pred = np.where(gate>=0.5,teacher.argmax(1),p.argmax(1))
        return pred,gate
