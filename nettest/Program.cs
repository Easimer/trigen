using System;
using Net.Easimer.Trigen;

static class Program
{
    static void Main(string[] args)
    {
        var parameters = new Parameters();
        var session = Session.Create(parameters);

        Console.ReadKey();
    }
}