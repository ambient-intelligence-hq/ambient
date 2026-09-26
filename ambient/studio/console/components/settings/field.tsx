import type { ReactNode } from "react";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

// A consistent form-row: label + optional description + control, with even
// rhythm. Keeps every settings field looking the same without pulling in a full
// form library for a handful of fields.
export function Field({
  label,
  htmlFor,
  description,
  children,
  className,
}: {
  label: string;
  htmlFor?: string;
  description?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("grid gap-2", className)}>
      <Label className="font-medium text-sm" htmlFor={htmlFor}>
        {label}
      </Label>
      {children}
      {description && (
        <p className="text-muted-foreground text-xs leading-relaxed">
          {description}
        </p>
      )}
    </div>
  );
}
