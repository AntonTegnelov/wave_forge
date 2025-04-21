use crate::buffers::rule_buffers::RuleBuffers;
use crate::error::GpuError;
use crate::gpu::device::setup_device;
use crate::gpu::sync::{create_rule_bind_group_layout, create_rule_bind_groups};
use crate::rules::AdjacencyRules;

pub struct WFC {
    device: wgpu::Device,
    queue: wgpu::Queue,
    rule_buffers: RuleBuffers,
    rule_bind_group: wgpu::BindGroup,
}

impl WFC {
    pub fn new(rules: AdjacencyRules) -> Result<Self, GpuError> {
        let (device, queue) = setup_device()?;

        // Create rule buffers
        let rule_buffers = RuleBuffers::new(&device, &rules)?;

        // Create bind group layout and bind group for rules
        let rule_bind_group_layout = create_rule_bind_group_layout(&device);
        let rule_bind_group =
            create_rule_bind_groups(&device, &rule_buffers, &rule_bind_group_layout)?;

        Ok(Self {
            device,
            queue,
            rule_buffers,
            rule_bind_group,
        })
    }
}
