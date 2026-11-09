import { AmbientAvatar } from "@/components/chat/ambient-mark";
import { Card, CardContent } from "@/components/ui/card";

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <main className="flex min-h-dvh w-full flex-col items-center justify-center bg-sidebar px-4 py-16">
      <div className="flex w-full max-w-sm flex-col gap-8">
        <div className="flex flex-col items-center gap-3 text-center">
          <AmbientAvatar className="size-14 rounded-2xl" />
          <div className="flex flex-col gap-1">
            <p className="text-xl font-semibold tracking-tight">Ambient</p>
            <p className="text-sm text-muted-foreground">
              Video engine for AI agents
            </p>
          </div>
        </div>
        <Card className="rounded-2xl border-border/50">
          <CardContent className="flex flex-col gap-1.5 p-6 sm:p-8">
            {children}
          </CardContent>
        </Card>
      </div>
    </main>
  );
}
