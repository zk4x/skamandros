/* Link script for the RISCV B data mover (`dm_b`) and the B executors.
 *
 * B's reset PC is hardwired to L1 offset 0 and it has no reset-PC override
 * (`tt_isa::tensix::Core::B`), so the host writes one `jal` at 0 that starts
 * the image here, as tt-metal starts BRISC (`generate_risc_startup_addr`,
 * `llrt/hal.cpp:126-149`). The image is not at 0: the card's CMFW newer than
 * 19.3 writes its throttle state to L1 offset 0x10 on every Tensix
 * (`tt_isa::dm::IMAGE_BASE`). LENGTH stops at 0xA800, where the first role
 * image (T0's slot, `tt_isa::l1::roles`) begins, so the mover and the three
 * role images can all be resident in one tile at once -- a link error, not a
 * clobbered T0 entry point, if it ever grows past that
 * (`tt_isa::dm::IMAGE_MAX`).
 */
MEMORY
{
  L1    (rwx) : ORIGIN = 0x00001000, LENGTH = 0x9800
  LOCAL (rw)  : ORIGIN = 0xFFB00000, LENGTH = 4K
}

INCLUDE sections.x
