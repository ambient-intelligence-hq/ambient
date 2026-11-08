import { connection } from "next/server";
import { Suspense } from "react";
import { isClosedAccess } from "@/lib/constants";
import { LoginForm } from "./login-form";

// Read at request time: the same image serves open and closed installs.
async function LoginFormForAccess() {
  await connection();
  return <LoginForm allowSignUp={!isClosedAccess()} />;
}

export default function Page() {
  return (
    <Suspense>
      <LoginFormForAccess />
    </Suspense>
  );
}
