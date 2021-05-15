using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Net.Easimer.Trigen
{
    public enum Status
    {
        OK = 0,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct Parameters
    {
    }

    public class Exception : System.Exception
    {
        public Status StatusCode { get; protected set; }

        public Exception(Status statusCode)
        {
            StatusCode = statusCode;
        }
    }

    public class Session : IDisposable
    {
        private IntPtr sessionHandle;

        protected Session(IntPtr sessionHandle)
        {
            this.sessionHandle = sessionHandle;
        }

        public void Dispose()
        {
            if(sessionHandle != IntPtr.Zero)
            {
                DestroySession(sessionHandle);
                sessionHandle = IntPtr.Zero;
            }
        }

        public static Session Create(Parameters param)
        {
            IntPtr handle;
            var status = CreateSession(out handle, ref param);
            if(status != Status.OK)
            {
                throw new Exception(status);
            }

            Debug.Assert(handle != IntPtr.Zero);

            return new Session(handle);
        }

        [DllImport("trigen.dll", EntryPoint = "Trigen_CreateSession")]
        private static extern Status CreateSession([Out] out IntPtr handle, [In] ref Parameters param);

        [DllImport("trigen.dll", EntryPoint = "Trigen_DestroySession")]
        private static extern Status DestroySession([In] IntPtr handle);
    }
}