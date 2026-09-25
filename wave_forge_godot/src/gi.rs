//! Which global illumination an instance the extension creates takes part in, by what it holds
//! (docs/architecture/engine-integration.md, "Content classes"): the ground and buildings are
//! static, so SDFGI and baked lighting take them in; grass and scattered props take none, being
//! many, small and swaying.

use godot::classes::RenderingServer;
use godot::classes::rendering_server::InstanceFlags;
use godot::prelude::*;

/// A content class's global illumination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gi {
    /// Taken in by SDFGI and baked lighting: `GeometryInstance3D`'s `GI_MODE_STATIC`.
    Static,
    /// Taken in by none: `GI_MODE_DISABLED`.
    Off,
}

impl Gi {
    /// The instance flags `USE_BAKED_LIGHT` and `USE_DYNAMIC_GI`, as
    /// `GeometryInstance3D::set_gi_mode` sets them for the same mode.
    const fn flags(self) -> (bool, bool) {
        match self {
            Self::Static => (true, false),
            Self::Off => (false, false),
        }
    }

    /// Sets a RenderingServer instance's flags to this.
    pub(crate) fn apply(self, instance: Rid) {
        let (baked, dynamic) = self.flags();
        let mut rendering = RenderingServer::singleton();
        rendering.instance_geometry_set_flag(instance, InstanceFlags::USE_BAKED_LIGHT, baked);
        rendering.instance_geometry_set_flag(instance, InstanceFlags::USE_DYNAMIC_GI, dynamic);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_gi_is_baked_light_without_dynamic_gi() {
        assert_eq!(Gi::Static.flags(), (true, false));
    }

    #[test]
    fn no_gi_is_neither() {
        assert_eq!(Gi::Off.flags(), (false, false));
    }
}
