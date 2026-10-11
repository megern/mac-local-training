"""Retain an endpoint Adam group between synchronous updates.

This is a scheduling optimization of existing optimizer offloading, not a new
optimizer. All gradients are evaluated before any group update. One group's
moments stay resident, and the next update begins with that same group.
"""
from .offloaded import OffloadedTrainingEngine
from .engine import _json_write


def endpoint_order(groups, completed_steps):
    if type(groups) is not int or groups < 1:
        raise ValueError('groups must be a positive integer')
    if type(completed_steps) is not int or completed_steps < 0:
        raise ValueError('completed_steps must be a nonnegative integer')
    order = list(range(groups))
    return order if completed_steps % 2 == 0 else order[::-1]


def cold_switch_counts(groups, steps):
    """Count evictions, excluding the first empty-cache activation."""
    endpoint_order(groups, 0)
    if type(steps) is not int or steps < 1:
        raise ValueError('steps must be a positive integer')
    return {'fixed': 0 if groups == 1 else groups * steps - 1,
            'endpoint': (groups - 1) * steps}


class EndpointTrainingEngine(OffloadedTrainingEngine):
    def _identity(self):
        identity = super()._identity()
        identity['schedule'] = 'endpoint-retained-grouped-adam-v1'
        return identity

    def update_group_order(self):
        return endpoint_order(len(self.groups), self.step)

    def update_once(self):
        order = self.update_group_order()
        record = super().update_once()
        record['group_update_order'] = order
        record['group'] = 'all_endpoint_retained'
        return record

    def run(self):
        report = super().run()
        report['strategy'] = 'endpoint_retained_grouped_adam'
        report['limitations'].append('Endpoint retention reduces state switches, not necessarily elapsed time; disk page caching and memory pressure affect results.')
        _json_write(self.output / 'report.json', report)
        return report
