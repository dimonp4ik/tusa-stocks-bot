import unittest
from chronological_entry_model import schema,vector,fit,probability

class EntryModelTests(unittest.TestCase):
    def test_unseen_category_does_not_extend_fitted_schema(self):
        train=[{'volume_ratio':1,'direction':'LONG','net_r':1}, {'volume_ratio':3,'direction':'SHORT','net_r':-1}]
        design=schema(train);before=repr(design)
        a=vector(train[0],*design)
        b=vector({'volume_ratio':999,'direction':'FUTURE_CATEGORY'},*design)
        self.assertEqual(len(a),len(b))
        self.assertEqual(repr(design),before)
        self.assertEqual(b[1],5)

    def test_model_learns_known_predictive_feature(self):
        rows=[{'volume_ratio':float(i%2),'net_r':1 if i%2 else -1} for i in range(40)]
        design=schema(rows);weights=fit(rows,design)
        low=probability(vector(rows[0],*design),weights)
        high=probability(vector(rows[1],*design),weights)
        self.assertLess(low,.2)
        self.assertGreater(high,.8)
