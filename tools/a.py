from typing import List, Set, Dict, Tuple, Optional
import nltk
import numpy as np
from difflib import SequenceMatcher
from scipy.optimize import linear_sum_assignment
from symbolic_solvers.fol_solver.fol_parser import FOL_Parser

class FOLTreeSimilarity:
    def __init__(self):
        self.parser = FOL_Parser()
    
    def tree_similarity(self, tree1: Optional[nltk.Tree], tree2: Optional[nltk.Tree]) -> float:
        """Calculate similarity between two parse trees, handling None cases"""
        # Handle None cases
        if tree1 is None and tree2 is None:
            return 1.0
        if tree1 is None or tree2 is None:
            return 0.0
            
        if isinstance(tree1, str) and isinstance(tree2, str):
            return SequenceMatcher(None, tree1.lower(), tree2.lower()).ratio()
        
        if isinstance(tree1, str) or isinstance(tree2, str):
            return 0.0
        
        if tree1.label() != tree2.label():
            return 0.0
        
        if len(tree1) == 1 and len(tree2) == 1 and isinstance(tree1[0], str) and isinstance(tree2[0], str):
            return SequenceMatcher(None, tree1[0].lower(), tree2[0].lower()).ratio()
        
        max_len = max(len(tree1), len(tree2))
        if max_len == 0:
            return 1.0
            
        similarities = np.zeros((len(tree1), len(tree2)))
        for i, child1 in enumerate(tree1):
            for j, child2 in enumerate(tree2):
                similarities[i,j] = self.tree_similarity(child1, child2)
        
        row_ind, col_ind = linear_sum_assignment(-similarities)
        matched_similarity = similarities[row_ind, col_ind].mean()
        size_penalty = min(len(tree1), len(tree2)) / max_len
        
        return matched_similarity * size_penalty

    def safe_parse(self, formula: str) -> Tuple[Optional[nltk.Tree], str]:
        """Safely parse a formula, returning both the tree and any error message"""
        try:
            tree = self.parser.parse_text_FOL_to_tree(formula)
            return tree, ""
        except Exception as e:
            return None, str(e)

    def extract_components(self, tree: Optional[nltk.Tree]) -> Tuple[Set[str], Set[str], Set[str]]:
        """Extract components from a tree, handling None case"""
        if tree is None:
            return set(), set(), set()
            
        predicates, variables, constants = set(), set(), set()
        
        def traverse(t):
            if isinstance(t, str):
                return
            
            if t.label() == 'PRED':
                predicates.add(t[0])
            elif t.label() == 'VAR':
                variables.add(t[0])
            elif t.label() == 'CONST':
                constants.add(t[0])
            
            for child in t:
                traverse(child)
                
        traverse(tree)
        return predicates, variables, constants

    def compare_problems(self, problem1: List[str], problem2: List[str]) -> dict:
        """Compare two FOL problems, handling parsing failures"""
        # Parse all formulas and track parsing failures
        parsed_results1 = [self.safe_parse(f) for f in problem1]
        parsed_results2 = [self.safe_parse(f) for f in problem2]
        
        trees1 = [result[0] for result in parsed_results1]
        trees2 = [result[0] for result in parsed_results2]
        
        # Calculate valid formula counts
        valid_trees1 = [t for t in trees1 if t is not None]
        valid_trees2 = [t for t in trees2 if t is not None]
        
        if not valid_trees1 or not valid_trees2:
            return {
                'error': 'No valid formulas to compare'
            }
        
        # Calculate similarity matrix for valid trees
        similarities = np.zeros((len(valid_trees1), len(valid_trees2)))
        for i, tree1 in enumerate(valid_trees1):
            for j, tree2 in enumerate(valid_trees2):
                similarities[i,j] = self.tree_similarity(tree1, tree2)
        
        # Find best matching between valid formulas
        row_ind, col_ind = linear_sum_assignment(-similarities)
        formula_matching = similarities[row_ind, col_ind].mean()
        
        # Extract components from valid trees
        all_pred1, all_var1, all_const1 = set(), set(), set()
        all_pred2, all_var2, all_const2 = set(), set(), set()
        
        for tree in valid_trees1:
            p, v, c = self.extract_components(tree)
            all_pred1.update(p)
            all_var1.update(v)
            all_const1.update(c)
            
        for tree in valid_trees2:
            p, v, c = self.extract_components(tree)
            all_pred2.update(p)
            all_var2.update(v)
            all_const2.update(c)
        
        # Calculate similarity scores
        result = {
            'predicate_similarity': self._set_similarity(all_pred1, all_pred2),
            'variable_similarity': self._set_similarity(all_var1, all_var2),
            'constant_similarity': self._set_similarity(all_const1, all_const2),
            'formula_matching': formula_matching,
            'size_similarity': min(len(valid_trees1), len(valid_trees2)) / max(len(valid_trees1), len(valid_trees2)),
        }
        
        return result

    def _set_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        if not set1 or not set2:
            return 0.0
        
        similarities = []
        for s1 in set1:
            best_sim = max(SequenceMatcher(None, s1.lower(), s2.lower()).ratio() 
                          for s2 in set2)
            similarities.append(best_sim)
            
        return sum(similarities) / len(similarities)

# Example usage
def demonstrate_error_handling():
    calculator = FOLTreeSimilarity()
    
    # Example with some invalid formulas
    problem2 = [
        "∀x (ClubMember(x) ∧ Perform(x, TalentShow(s)) ∧ SchoolEvent(s) → (Attend(x, SchoolEvent(s))))",
        "∀x (ClubMember(x) → (Perform(x, TalentShow(s)) ⊕ Inactive(x)))",
        "∀x (ClubMember(x) ∧ Chaperone(x, HighSchoolDance(s)) → ¬Student(x, School(s)))",
        "∀x (ClubMember(x) ∧ Inactive(x) → Chaperone(x, HighSchoolDance(s)))",
        "∀x (ClubMember(x) ∧ YoungChildOrTeenager(x) ∧ WishToFurtherAcademics(x) → Student(x, School(s)))",
        "(ClubMember(bonnie) ∧ (Attend(bonnie, SchoolEvent(s)) ∧ Student(bonnie, School(s))) ∨ ¬(Attend(bonnie, SchoolEvent(s)) ∧ Student(bonnie, School(s))))"
    ]
    
    problem3 = [
        "∀x (Inclub(x) ∧ Perform(x, school) → (Attend(x, school) ∧ Engaged(x, school)))",
        "∀x (Inclub(x) → (Perform(x, school) ⊕ Inactive(x) ∧ Disinterested(x)))",
        "∀x (Inclub(x) ∧ Chaperone(x, school) → ¬Student(x, school))",
        "∀x (Inclub(x) ∧ Inactive(x) ∧ Disinterested(x) → Chaperone(x, school))",
        "∀x (Childorteenager(x) ∧ Inclub(x) ∧ Wishtofurtheredu(x) → Student(x, school))",
        "Inclub(bonnie) ∧ ((Attend(bonnie, school) ∧ Engaged(bonnie, school)) ∨ ¬((Attend(bonnie, school) ∧ Engaged(bonnie, school))))"
    ]
    
    problem1 = [
      "∀x (InThisClub(x) ∧ PerformOftenIn(x, schoolTalentShow) → Attend(x, schoolEvent) ∧ VeryEngagedWith(x, schoolEvent))",
      "∀x (InThisClub(x) → PerformOftenIn(x, schoolTalentShow) ⊕ (InActive(x) ∧ Disinterested(x) ∧ MemberOf(x, community)))",
      "∀x (InThisClub(x) ∧ Chaperone(x, highSchoolDance) → ¬(Studen(x) ∧ AttendSchool(x)))",
      "∀x (InThisClub(x) ∧ (InActive(x) ∧ Disinterested(x) ∧ MemberOf(x, community)) → Chaperone(x, highSchoolDances))",
      "∀x (InThisClub(x) ∧ (YoungChildren(x) ⊕ Teenager(x)) ∧ WishToFurther(x, academicCareer)) → Studen(x) ∧ AttendSchool(x))",
      "InThisClub(bonnie) ∧ ¬((Attend(x, schoolEvent) ∧ VeryEngagedWith(bonnie, schoolEvent)) ⊕ (Studen(bonne) ∧ AttendSchool(bonnie)))"
    ]
    
    result = calculator.compare_problems(problem1, problem3)
    
    print("Comparison Results:")
    for key, value in result.items():
        print(f"\n{key}:")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"  {value}")

if __name__ == "__main__":
    demonstrate_error_handling()