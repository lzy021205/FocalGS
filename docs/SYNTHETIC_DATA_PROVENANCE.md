# Synthetic Scene Provenance and Redistribution Status

Date: 2026-10-13

This record separates source-code licensing, Blender scene-asset licensing,
and permission to redistribute FocalGS rerendered images. Those are independent
rights. In particular, the MIT license in the Deblur-NeRF code repository does
not grant rights to every Blender asset used by its datasets.

## Evidence summary

| FocalGS scene | Best-supported source | Publisher or creator | Asset license evidence | Release decision |
| --- | --- | --- | --- | --- |
| Tanabata | Custom multi-focal rerender of the Deblur-NeRF `tanabata` scene | Deblur-NeRF dataset by Ma et al.; underlying asset creator unknown | Deblur-NeRF publishes data and raw Blender files, but no asset-specific license was found. A later public license inventory also leaves this scene's author/license fields blank. | FocalGS public release authorized; upstream asset-license record incomplete |
| Cozyroom | Custom multi-focal rerender of the Deblur-NeRF `cozyroom` scene | Deblur-NeRF dataset by Ma et al.; underlying asset creator unknown | No asset-specific redistribution license found | FocalGS public release authorized; upstream asset-license record incomplete |
| Pool | Custom multi-focal rerender of the Deblur-NeRF `pool` scene | Deblur-NeRF dataset by Ma et al.; underlying asset creator unknown | No asset-specific redistribution license found | FocalGS public release authorized; upstream asset-license record incomplete |
| Factory | Custom multi-focal rerender of the Deblur-NeRF `factory` scene | Deblur-NeRF dataset by Ma et al.; underlying asset creator unknown | No asset-specific redistribution license found | FocalGS public release authorized; upstream asset-license record incomplete |
| Building | Local Blender scene named `buildings`; no public source package identified | Unknown | No creator, source URL, or license recovered | FocalGS public release authorized; upstream source/license record incomplete |
| Livingroom | Strong visual/name match to BlendSwap **Bachelor's quarters**, blend 17574 | Rendars | BlendSwap currently labels the asset CC0. The local `.blend` file or source hash is absent, so the exact derivation still requires author attestation. | FocalGS public release authorized; exact source attribution remains unverified |
| Ship | Local icy-cliff/sailing-ship scene; no public source package identified | Unknown | No creator, source URL, or license recovered | FocalGS public release authorized; upstream source/license record incomplete |

The four Deblur-NeRF scene names and raw Blender release are documented by the
[official project page](https://limacv.github.io/deblurnerf/) and
[official repository](https://github.com/limacv/Deblur-NeRF). The repository's
[MIT license](https://github.com/limacv/Deblur-NeRF/blob/master/LICENSE) covers
its software; neither that file nor the project page states a single license
for all third-party Blender assets.

The supplementary material for
[Exploiting Deblurring Networks for Radiance Fields](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Choi_Exploiting_Deblurring_Networks_CVPR_2025_supplemental.pdf)
contains an asset-license inventory. Its Deblur-NeRF rows for Cozyroom,
Factory, Pool, and Tanabata leave author, license, and source-link fields blank.
This is corroborating evidence only; it is not a grant of rights.

The Livingroom candidate is
[Bachelor's quarters by Rendars](https://blendswap.com/blend/17574), whose
current BlendSwap page identifies the license as CC0. Because the exact source
`.blend` and its checksum were not preserved in the research archive, the
authors should confirm that this was the asset actually rendered before relying
on that license.

## Authorization and license scope

The project release authority has authorized public redistribution of the seven
frozen FocalGS rendered archives. The source audit above is retained so that
this publication approval is not mistaken for proof of ownership or a new
license grant for upstream Blender assets, textures, or models. Where an exact
asset-specific license was not recovered, no standard repository-wide data
license is asserted on its behalf.
