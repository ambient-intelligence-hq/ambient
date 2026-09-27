import { cn } from "@/lib/utils";

// FNV-1a: small, fast, and spreads similar seeds (guest-1727…, guest-1728…)
// into very different patterns.
function hash32(seed: string): number {
  let h = 0x81_1c_9d_c5;
  for (let i = 0; i < seed.length; i += 1) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 0x01_00_01_93);
  }
  return h >>> 0;
}

/**
 * A GitHub-style identicon: a 5×5 grid, mirrored left-to-right, in one colour
 * picked from the seed. The same seed always draws the same avatar, so a user
 * (guest or signed in) keeps a recognisable face without uploading anything.
 */
export function Identicon({
  seed,
  className,
}: {
  seed: string;
  className?: string;
}) {
  const h = hash32(seed);
  const hue = (h >>> 15) % 360;
  const cells: { x: number; y: number }[] = [];
  // 15 bits fill the left three columns; columns 3–4 mirror columns 1–0.
  for (let col = 0; col < 3; col += 1) {
    for (let row = 0; row < 5; row += 1) {
      if ((h >>> (col * 5 + row)) & 1) {
        cells.push({ x: col, y: row });
        if (col < 2) {
          cells.push({ x: 4 - col, y: row });
        }
      }
    }
  }

  return (
    <svg
      aria-hidden="true"
      className={cn(
        "shrink-0 rounded-md bg-neutral-100 ring-1 ring-sidebar-border/50 dark:bg-neutral-800",
        className
      )}
      shapeRendering="crispEdges"
      viewBox="-0.5 -0.5 6 6"
    >
      {cells.map(({ x, y }) => (
        <rect
          fill={`oklch(0.64 0.14 ${hue})`}
          height={1}
          key={`${x}-${y}`}
          width={1}
          x={x}
          y={y}
        />
      ))}
    </svg>
  );
}
