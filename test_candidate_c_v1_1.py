"""Deterministic checks only. Reference calculations do not reuse metric formulas."""
from fractions import Fraction as F
import itertools
import math
import unittest
import numpy as np
import eviconsist_candidate_c_v1_1 as c

def direct(p):
    # Separate nested enumeration of probabilities, not candidate coefficient arrays.
    values=[]
    for model in ('e','c'):
        tp=flags=positive=F(0)
        for task in range(3):
            for y,e,combined in itertools.product(range(2),repeat=3):
                probability=p[8*task+4*y+2*e+combined]/3
                decision=e if model=='e' else combined
                tp+=probability*y*decision;flags+=probability*decision;positive+=probability*y
        values.append((tp/flags if flags else None,tp/positive if positive else None))
    return values[1][0]-values[0][0],values[1][1]-values[0][1]

class MathematicalChecks(unittest.TestCase):
    def test_phase_one_proves_infeasible_branch(self):
        audit=c.Audit()
        with self.assertRaises(c.CertifiedEmptyNode):
            c.linear_bound([F(1)],[[F(1)]],[F(-1)],[],[],[F(0)],[F(1)],audit,'known_empty')
        cert=audit.certificates[-1]
        self.assertGreater(F(cert['lower_exact']),0)
        phase=c.phase_one_model([[F(1)]],[F(-1)],[],[],[F(0)],[F(1)])
        obj,A,b,H,d,lo,hi=phase
        self.assertEqual(c.exact_dual_lower(obj,A,b,H,d,lo,hi,list(map(F,cert['inequality_duals'])),[]),F(cert['lower_exact']))
    def test_allocation(self):
        self.assertEqual(48*c.TAIL,F(1,40))

    def test_n_one_hand_limits(self):
        l,u=c.cp_bounds(0,1); self.assertEqual(l,0); self.assertGreaterEqual(u,1-c.TAIL)
        l,u=c.cp_bounds(1,1); self.assertLessEqual(l,c.TAIL);self.assertEqual(u,1)

    def test_tail_enclosures_against_exact_finite_sums(self):
        for n in range(1,6):
            for k in range(n+1):
                for p in (F(1,100),F(1,2),F(99,100)):
                    for sf in (False,True):
                        js=range(k,n+1) if sf else range(k+1)
                        exact=sum((F(math.comb(n,j))*p**j*(1-p)**(n-j) for j in js),F(0))
                        lo,hi=c.finite_tail_enclosure(n,k,p,sf)
                        self.assertLessEqual(F(lo),exact);self.assertGreaterEqual(F(hi),exact)

    def test_small_exact_binomial_coverage_enumeration(self):
        for n in range(1,6):
            for p in (F(0),F(1,100),F(1,2),F(99,100),F(1)):
                covered=F(0)
                for k in range(n+1):
                    l,u=c.cp_bounds(k,n)
                    if l<=p<=u: covered+=F(math.comb(n,k))*p**k*(1-p)**(n-k)
                self.assertGreaterEqual(covered,1-2*c.TAIL)

    def test_weighted_metric_against_direct_response_enumeration(self):
        counts=[[10,0,0,0,0,2,0,8],[0,4,0,0,0,0,4,2],[4,0,4,0,0,0,0,2]]
        r=c.from_counts(counts)
        self.assertEqual(c.contrasts(r.anchor),direct(r.anchor))
        # Unequal task counts must not become raw pooled weights.
        pooled=[F(sum(row[j] for row in counts),sum(map(sum,counts))) for j in range(8)]*3
        self.assertNotEqual(c.contrasts(r.anchor),direct(pooled))

    def test_conditional_two_by_two_and_degenerate_projection(self):
        point=[F(v,100) for v in [0,0,0,0,10,20,5,65]]*3
        r=c.Region(point,point,point)
        self.assertEqual(direct(point),(F(0),F(15,100)))
        audit=c.Audit();p=c.precision_projection(r,audit);rec=c.recall_projection(r,audit)
        self.assertEqual(p['status'],'certified');self.assertEqual(rec['status'],'certified')
        self.assertLessEqual(p['lower'],0);self.assertGreaterEqual(p['upper'],0)
        self.assertLessEqual(rec['lower'],.15);self.assertGreaterEqual(rec['upper'],.15)

    def test_arbitrary_duals_valid_against_independent_vertex_enumeration(self):
        # Square clipped by x+y<=1, vertices known by hand; duals deliberately nonoptimal.
        A=[[F(1),F(1)]]; b=[F(1)]; H=[];d=[];lo=[F(0)]*2;hi=[F(1)]*2
        vertices=[(F(0),F(0)),(F(1),F(0)),(F(0),F(1))]
        for obj in ([F(2),F(-3)],[F(-1),F(-1)],[F(1,7),F(2,3)]):
            truth=min(sum(a*x for a,x in zip(obj,p)) for p in vertices)
            for y in (F(-100),F(-1),F(0),F(2)):
                bound=c.exact_dual_lower(obj,A,b,H,d,lo,hi,[y],[])
                self.assertLessEqual(bound,truth)

    def test_denominator_ranges_against_simplex_vertex_enumeration(self):
        # Nontrivial box with four active categories, all other probabilities fixed zero.
        p=[F(1,4),F(1,4),F(1,4),F(1,4),0,0,0,0]*3
        l=[F(1,10)]*4+[F(0)]*4;u=[F(4,10)]*4+[F(0)]*4
        r=c.Region(l*3,u*3,p)
        for coeff in (c.DC,c.DE):
            low,high=r.denominator_range(coeff)
            self.assertEqual(low,F(1,5));self.assertEqual(high,F(4,5))

    def test_boundary_probabilities_and_unobserved_cells_remain_uncertain(self):
        r=c.from_counts([[16,0,0,4,2,0,0,18]]*3)
        for t in range(3):
            for j in (1,2,5,6):
                self.assertEqual(r.lower[8*t+j],0);self.assertGreater(r.upper[8*t+j],0)
        # Two feasible explicit distributions with opposite recall disagreements.
        for sign in (-1,1):
            p=r.anchor.copy();j=5 if sign>0 else 6
            for t in range(3):
                epsilon=min(r.upper[8*t+j],r.anchor[8*t+7]-r.lower[8*t+7])/2
                p[8*t+j]+=epsilon;p[8*t+7]-=epsilon
            self.assertTrue(r.feasible(p));self.assertGreater(sign*c.contrasts(p)[1],0)

    def test_no_flags_and_no_positives_no_success(self):
        for counts in ([[20,0,0,0,20,0,0,0]]*3,[[10,10,10,10,0,0,0,0]]*3):
            r=c.from_counts(counts);audit=c.Audit()
            endpoint=c.precision_projection(r,audit) if counts[0][1]==0 else c.recall_projection(r,audit)
            self.assertEqual(endpoint['status'],'non_estimable')

    def test_absent_task_empty_region_invalid_counts(self):
        for counts in ([[1]*8]*2,[[0]*8,[1]*8,[1]*8],[[1]*7]*3,[[1.0]*8]*3):
            with self.assertRaises(c.InvalidDesign): c.from_counts(counts)
        with self.assertRaises(c.InvalidDesign):c.Region([F(1,5)]*24,[F(1,4)]*24,[F(1,8)]*24)

    def test_duplicated_group_and_invalid_binary_rejected(self):
        base=dict(group='a',task='QA',y=0,e=1,c=0)
        with self.assertRaises(c.InvalidDesign):c.count_records([base,base])
        with self.assertRaises(c.InvalidDesign):c.count_records([dict(base,y=2)])

    def test_lp_certificate_replay_and_outward_float(self):
        A=[[F(1),F(1)]];b=[F(1)];H=[];d=[];lo=[F(0)]*2;hi=[F(1)]*2;obj=[F(2),F(-3)]
        audit=c.Audit();bound,x=c.linear_bound(obj,A,b,H,d,lo,hi,audit,'unit_simplex')
        cert=audit.certificates[-1]
        replay=c.exact_dual_lower(obj,A,b,H,d,lo,hi,list(map(F,cert['inequality_duals'])),[])
        self.assertEqual(replay,bound);self.assertLessEqual(bound,-3)
        for q in (F(1,3),F(-1,7),F(1,2**100)):
            self.assertLessEqual(F(c.outward(q,True)),q);self.assertGreaterEqual(F(c.outward(q,False)),q)

if __name__=='__main__':unittest.main(verbosity=2)
