' External iLogic rule: update all named Inventor sauger parameters.
' Expected arguments: s1_x, s1_z, s2_x, s2_z, ...

If RuleArguments.Count = 0 OrElse RuleArguments.Count Mod 2 <> 0 Then
    Throw New Exception("Expected paired sauger position arguments.")
End If

Dim saugerCount As Integer = RuleArguments.Count \ 2
For saugerIndex As Integer = 1 To saugerCount
    For Each axis As String In New String() {"x", "z"}
        Dim parameterName As String = "s" & saugerIndex & "_" & axis
        If Not RuleArguments.Exists(parameterName) Then
            Throw New Exception("Missing iLogic argument: " & parameterName)
        End If

        Parameter(parameterName) = CDbl(RuleArguments.Value(parameterName))
    Next
Next

InventorVb.DocumentUpdate()
