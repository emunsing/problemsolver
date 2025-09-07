#!/usr/bin/env python3
"""
Pareto frontier metrics for multi-objective optimization.

This module provides different definitions of Pareto optimality and dominance
for use in optimizer evaluation and comparison.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import numpy as np
from problemsolver.utils import Performance

# Type alias for points
Point = Tuple[float, float]


# --- small helper for convex hull (monotone chain lower hull) ---
def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def _lower_convex_hull(points: List[Point]) -> List[Point]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts[:]
    lower = []
    for p in pts:
        # <= 0 removes collinear middle points (so "on-hull" is NOT a strict improvement)
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    return lower


class ParetoMetric(ABC):
    """Abstract base class for different Pareto optimality definitions."""
    
    @classmethod
    @abstractmethod
    def is_dominated(cls, candidate: Performance, existing_points: List[Performance]) -> bool:
        """Check if a candidate point is dominated by any existing points."""
        pass
    
    @classmethod
    @abstractmethod
    def get_frontier(cls, points: List[Performance]) -> List[Performance]:
        """Get the Pareto frontier from a list of points."""
        pass
    
    @classmethod
    @abstractmethod
    def is_improvement(cls, new_point: Performance, existing_frontier: List[Performance], rtol: float = 0.0) -> bool:
        """Check if a new point represents a Pareto improvement over the existing frontier."""
        pass
    
    @classmethod
    @abstractmethod
    def analyze_gaps(cls, new_point: Performance, existing_frontier: List[Performance]) -> Dict:
        """Analyze how close a new point is to breaking through the Pareto frontier."""
        pass


class StrictDominanceParetoMetric(ParetoMetric):
    """
    Traditional Pareto optimality using strict dominance.
    
    A point dominates another if it is at least as good in all metrics 
    and strictly better in at least one metric.
    """
    
    @classmethod
    def is_dominated(cls, candidate: Performance, existing_points: List[Performance]) -> bool:
        """Check if candidate is dominated by any existing point."""
        if not candidate.is_successful() or candidate.log_rel_error is None or candidate.time_elapsed is None:
            return False
        
        for existing in existing_points:
            if not existing.is_successful() or existing.log_rel_error is None or existing.time_elapsed is None:
                continue
                
            # Check if existing dominates candidate
            if (existing.log_rel_error <= candidate.log_rel_error and 
                existing.time_elapsed <= candidate.time_elapsed and
                (existing.log_rel_error < candidate.log_rel_error or 
                 existing.time_elapsed < candidate.time_elapsed)):
                return True
        return False
    
    @classmethod
    def get_frontier(cls, points: List[Performance]) -> List[Performance]:
        """Compute the Pareto frontier using strict dominance."""
        if not points:
            return []
        
        frontier = []
        
        for candidate in points:
            # Only consider successful results for Pareto frontier
            if not candidate.is_successful() or candidate.log_rel_error is None or candidate.time_elapsed is None:
                continue
                
            if not cls.is_dominated(candidate, frontier):
                # Remove any existing frontier points that are dominated by this candidate
                frontier = [point for point in frontier if not cls.is_dominated(point, [candidate])]
                frontier.append(candidate)
        
        return frontier
    
    @classmethod
    def is_improvement(cls, new_point: Performance, existing_frontier: List[Performance], rtol: float = 0.0) -> bool:
        """Check if new point represents a Pareto improvement."""
        if not new_point.is_successful() or new_point.log_rel_error is None or new_point.time_elapsed is None:
            return False
        adjusted_frontier = [Performance(
                    name=existing.name,
                    log_rel_error=existing.log_rel_error + rtol * abs(existing.log_rel_error),
                    time_elapsed=(1+rtol) * existing.time_elapsed,
                ) for existing in existing_frontier]
        return not cls.is_dominated(new_point, adjusted_frontier)
    
    @classmethod
    def analyze_gaps(cls, new_point: Performance, existing_frontier: List[Performance]) -> Dict:
        """Analyze gaps for strict dominance Pareto metric."""
        if not existing_frontier:
            return {
                'closest_breakthrough_distance': 0.0,
                'error_breakthrough_gap': 0.0,
                'time_breakthrough_gap': 0.0,
                'best_error': float('inf'),
                'best_time': float('inf'),
                'needs_error_improvement': False,
                'needs_time_improvement': False,
                'closest_breakthrough_dimension': 'none'
            }
        
        # Find the best values in each dimension for reference
        best_error = min(point.log_rel_error for point in existing_frontier if point.log_rel_error is not None)
        best_time = min(point.time_elapsed for point in existing_frontier if point.time_elapsed is not None)
        
        # Check if the new point is already an improvement (not dominated)
        if not cls.is_dominated(new_point, existing_frontier):
            return {
                'closest_breakthrough_distance': 0.0,
                'error_breakthrough_gap': 0.0,
                'time_breakthrough_gap': 0.0,
                'best_error': best_error,
                'best_time': best_time,
                'needs_error_improvement': False,
                'needs_time_improvement': False,
                'closest_breakthrough_dimension': 'none'
            }
        
        # The new point is dominated, so we need to find how much improvement is needed
        # to make it not dominated by any frontier point
        
        # For each frontier point that dominates the new point, calculate the minimum
        # improvement needed to break the dominance
        error_improvements_needed = []
        time_improvements_needed = []
        
        for frontier_point in existing_frontier:
            if (not frontier_point.is_successful() or 
                frontier_point.log_rel_error is None or 
                frontier_point.time_elapsed is None):
                continue
            
            # Check if this frontier point dominates the new point
            if (frontier_point.log_rel_error <= new_point.log_rel_error and 
                frontier_point.time_elapsed <= new_point.time_elapsed and
                (frontier_point.log_rel_error < new_point.log_rel_error or 
                 frontier_point.time_elapsed < new_point.time_elapsed)):
                
                # This frontier point dominates the new point
                # Calculate how much we need to improve to break this dominance
                error_gap = new_point.log_rel_error - frontier_point.log_rel_error
                time_gap = new_point.time_elapsed - frontier_point.time_elapsed
                
                # To break dominance, we need to be better in at least one dimension
                # where we're currently worse or equal
                if error_gap >= 0 and time_gap >= 0:
                    # We're worse or equal in both dimensions
                    # We need to improve in at least one dimension
                    error_improvements_needed.append(error_gap + 1e-10)  # Small epsilon to break equality
                    time_improvements_needed.append(time_gap + 1e-10)
                elif error_gap >= 0:
                    # We're worse in error, equal or better in time
                    # We need to improve error to break dominance
                    error_improvements_needed.append(error_gap + 1e-10)
                elif time_gap >= 0:
                    # We're worse in time, equal or better in error  
                    # We need to improve time to break dominance
                    time_improvements_needed.append(time_gap + 1e-10)
        
        # Find the minimum improvement needed in each dimension
        min_error_improvement = min(error_improvements_needed) if error_improvements_needed else 0.0
        min_time_improvement = min(time_improvements_needed) if time_improvements_needed else 0.0
        
        # Determine which dimension is closest to breakthrough
        if min_error_improvement < min_time_improvement:
            closest_dimension = 'error'
            closest_breakthrough_distance = min_error_improvement
        elif min_time_improvement < min_error_improvement:
            closest_dimension = 'time'
            closest_breakthrough_distance = min_time_improvement
        else:
            closest_dimension = 'tie'
            closest_breakthrough_distance = min_error_improvement
        
        return {
            'closest_breakthrough_distance': closest_breakthrough_distance,
            'error_breakthrough_gap': min_error_improvement,
            'time_breakthrough_gap': min_time_improvement,
            'best_error': best_error,
            'best_time': best_time,
            'needs_error_improvement': min_error_improvement > 0,
            'needs_time_improvement': min_time_improvement > 0,
            'closest_breakthrough_dimension': closest_dimension
        }


class ConvexHullParetoMetric(ParetoMetric):
    """
    Convex hull-based Pareto optimality.
    
    A point is considered Pareto optimal if it lies on the convex hull
    of the objective space. This is a more relaxed definition that considers
    the convex combination of existing points.
    """
    
    @classmethod
    def is_dominated(cls, candidate: Performance, existing_points: List[Performance]) -> bool:
        """Check if candidate is dominated by the convex hull of existing points."""
        if not candidate.is_successful() or candidate.log_rel_error is None or candidate.time_elapsed is None:
            return False
        
        if not existing_points:
            return False
        
        # Get valid existing points
        valid_points = [p for p in existing_points 
                       if p.is_successful() and p.log_rel_error is not None and p.time_elapsed is not None]
        
        if not valid_points:
            return False
        
        # Convert to points for convex hull computation
        candidate_point = (candidate.log_rel_error, candidate.time_elapsed)
        existing_points_list = [(p.log_rel_error, p.time_elapsed) for p in valid_points]
        
        # Add the candidate point to the set
        all_points = existing_points_list + [candidate_point]
        
        # Compute convex hull
        hull_points = _lower_convex_hull(all_points)
        
        # Check if candidate is on the hull boundary
        is_on_hull = candidate_point in hull_points
        
        # If candidate is on the hull, it's not dominated
        # If candidate is inside the hull, it's dominated
        return not is_on_hull
            
    
    @classmethod
    def get_frontier(cls, points: List[Performance]) -> List[Performance]:
        """Compute the Pareto frontier using convex hull."""
        if not points:
            return []
        
        # Get valid points
        valid_points = [p for p in points 
                       if p.is_successful() and p.log_rel_error is not None and p.time_elapsed is not None]
        
        if not valid_points:
            return []
        
        if len(valid_points) == 1:
            return valid_points
        
        # Convert to points for convex hull computation
        points_list = [(p.log_rel_error, p.time_elapsed) for p in valid_points]
        
        # Compute convex hull
        hull_points = _lower_convex_hull(points_list)
        
        # Return performance objects that are on the hull
        hull_set = set(hull_points)
        return [p for p in valid_points if (p.log_rel_error, p.time_elapsed) in hull_set]

        
    @classmethod
    def is_improvement(cls, new_point: Performance, existing_frontier: List[Performance], rtol: float = 0.0) -> bool:
        """Check if new point represents a Pareto improvement using convex hull."""
        if not new_point.is_successful() or new_point.log_rel_error is None or new_point.time_elapsed is None:
            return False
        
        # Apply tolerance to existing points
        adjusted_points = [Performance(name=p.name, log_rel_error=p.log_rel_error + rtol * abs(p.log_rel_error), time_elapsed=(1+rtol) * p.time_elapsed) for p in existing_frontier]
        
        # Check if new point is dominated by the convex hull
        return not cls.is_dominated(new_point, adjusted_points)
    
    @classmethod
    def analyze_gaps(cls, new_point: Performance, existing_frontier: List[Performance]) -> Dict:
        """Analyze gaps for convex hull Pareto metric."""
        if not existing_frontier:
            return {
                'closest_breakthrough_distance': 0.0,
                'error_breakthrough_gap': 0.0,
                'time_breakthrough_gap': 0.0,
                'best_error': float('inf'),
                'best_time': float('inf'),
                'needs_error_improvement': False,
                'needs_time_improvement': False,
                'closest_breakthrough_dimension': 'none'
            }
        
        # Get valid points
        valid_points = [p for p in existing_frontier 
                       if p.is_successful() and p.log_rel_error is not None and p.time_elapsed is not None]
        
        if not valid_points:
            return {
                'closest_breakthrough_distance': 0.0,
                'error_breakthrough_gap': 0.0,
                'time_breakthrough_gap': 0.0,
                'best_error': float('inf'),
                'best_time': float('inf'),
                'needs_error_improvement': False,
                'needs_time_improvement': False,
                'closest_breakthrough_dimension': 'none'
            }
        
        # Find the best values in each dimension for reference
        best_error = min(p.log_rel_error for p in valid_points)
        best_time = min(p.time_elapsed for p in valid_points)
        
        # Check if the new point is already an improvement (not dominated by convex hull)
        if not cls.is_dominated(new_point, valid_points):
            return {
                'closest_breakthrough_distance': 0.0,
                'error_breakthrough_gap': 0.0,
                'time_breakthrough_gap': 0.0,
                'best_error': best_error,
                'best_time': best_time,
                'needs_error_improvement': False,
                'needs_time_improvement': False,
                'closest_breakthrough_dimension': 'none'
            }
        
        # The new point is dominated by the convex hull, so we need to find how much
        # improvement is needed to make it not dominated
        
        # For convex hull, we'll use a simplified approach: find the minimum distance
        # to any point on the hull boundary
        distances = []
        error_gaps = []
        time_gaps = []
        
        for frontier_point in valid_points:
            error_gap = max(0, new_point.log_rel_error - frontier_point.log_rel_error)
            time_gap = max(0, new_point.time_elapsed - frontier_point.time_elapsed)
            
            # Euclidean distance in objective space
            distance = np.sqrt(error_gap**2 + time_gap**2)
            distances.append(distance)
            error_gaps.append(error_gap)
            time_gaps.append(time_gap)
        
        # Find the closest frontier point
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        min_error_gap = error_gaps[min_distance_idx]
        min_time_gap = time_gaps[min_distance_idx]
        
        # Determine which dimension is closer to breakthrough
        if min_error_gap < min_time_gap:
            closest_dimension = 'error'
            closest_breakthrough_distance = min_error_gap
        elif min_time_gap < min_error_gap:
            closest_dimension = 'time'
            closest_breakthrough_distance = min_time_gap
        else:
            closest_dimension = 'tie'
            closest_breakthrough_distance = min_error_gap
        
        return {
            'closest_breakthrough_distance': closest_breakthrough_distance,
            'error_breakthrough_gap': min_error_gap,
            'time_breakthrough_gap': min_time_gap,
            'best_error': best_error,
            'best_time': best_time,
            'needs_error_improvement': min_error_gap > 0,
            'needs_time_improvement': min_time_gap > 0,
            'closest_breakthrough_dimension': closest_dimension
        }
