import { ArrowLeftIcon } from "lucide-react";
import Link from "next/link";

export function BackLink() {
  return (
    <Link
      className="fixed top-6 left-6 flex items-center gap-1.5 text-[13px] text-muted-foreground transition-colors hover:text-foreground"
      href="/"
    >
      <ArrowLeftIcon className="size-3.5" />
      Back
    </Link>
  );
}
