"use client";

import { ChevronUp } from "lucide-react";
import { useRouter } from "next/navigation";
import type { User } from "next-auth";
import { signOut, useSession } from "next-auth/react";
import { useTheme } from "next-themes";
import { useCallback } from "react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { guestRegex } from "@/lib/constants";
import { LoaderIcon } from "./icons";
import { Identicon } from "./identicon";
import { toast } from "./toast";

export function SidebarUserNav({ user }: { user: User }) {
  const router = useRouter();
  const { data, status } = useSession();
  const { setTheme, resolvedTheme } = useTheme();

  const isGuest = guestRegex.test(data?.user?.email ?? "");
  const handleThemeSelect = useCallback(() => {
    setTheme(resolvedTheme === "dark" ? "light" : "dark");
  }, [resolvedTheme, setTheme]);

  const handleAuthClick = useCallback(() => {
    if (status === "loading") {
      toast({
        description: "Checking authentication status, please try again!",
        type: "error",
      });

      return;
    }

    if (isGuest) {
      router.push("/login");
    } else {
      signOut({
        redirectTo: "/",
      });
    }
  }, [isGuest, router, status]);

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            {status === "loading" ? (
              <SidebarMenuButton className="h-10 justify-between rounded-lg bg-transparent text-sidebar-foreground/50 transition-colors duration-150 data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground">
                <div className="flex flex-row items-center gap-2">
                  <div className="size-6 animate-pulse rounded-full bg-sidebar-foreground/10" />
                  <span className="animate-pulse rounded-md bg-sidebar-foreground/10 text-transparent text-[13px]">
                    Loading...
                  </span>
                </div>
                <div className="animate-spin text-sidebar-foreground/50">
                  <LoaderIcon />
                </div>
              </SidebarMenuButton>
            ) : (
              <SidebarMenuButton
                className="h-8 px-2 rounded-lg bg-transparent text-sidebar-foreground/70 transition-colors duration-150 hover:text-sidebar-foreground data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
                data-testid="user-nav-button"
              >
                <Identicon
                  className="size-5"
                  seed={user.id ?? user.email ?? "guest"}
                />
                <span className="truncate text-[13px]" data-testid="user-email">
                  {isGuest ? "Guest" : user?.email}
                </span>
                <ChevronUp className="ml-auto size-3.5 text-sidebar-foreground/50" />
              </SidebarMenuButton>
            )}
          </DropdownMenuTrigger>
          <DropdownMenuContent
            className="w-(--radix-popper-anchor-width) rounded-lg border border-border/60 bg-card/95 backdrop-blur-xl shadow-[var(--shadow-float)]"
            data-testid="user-nav-menu"
            side="top"
          >
            <DropdownMenuItem
              className="cursor-pointer text-[13px]"
              data-testid="user-nav-item-theme"
              onSelect={handleThemeSelect}
            >
              {`Toggle ${resolvedTheme === "light" ? "dark" : "light"} mode`}
            </DropdownMenuItem>
            {/* Login is off for now: guests get no auth item; signed-in users
                can still sign out. */}
            {isGuest ? null : (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild data-testid="user-nav-item-auth">
                  <button
                    className="w-full cursor-pointer text-[13px]"
                    onClick={handleAuthClick}
                    type="button"
                  >
                    Sign out
                  </button>
                </DropdownMenuItem>
              </>
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
